import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class AutoGym:
  def __init__ (self, dataset_path, target_colomn):
    path = dataset_path
    df = pd.read_csv(path)
    target = target_colomn
    self.mode = "flexible"          # По умолчанию
    self.logs = []                  # Запись всего, что проиходит на сервере
    self.current_stage = "EDA"      # Для режима Fixed Transitions [EDA, Features, Train]
    self.max_attempts = 5           # Лимит попыток для режима Repeated
    self.attempts_made = 0          # Текущий счетчик попыток

    trainval, test = train_test_split(df, test_size=0.2, random_state=42)
    train, val = train_test_split(trainval, test_size=0.2, random_state=42)

    self.train_data = train
    self.val_x = val.drop(columns=[target])
    self.val_y = val[target]
    self.private_x = test.drop(columns=[target])
    self.private_y = test[target]

    self.candidates = {}

  def trigger_fallback(self):
        best_id = max(self.candidates, key=lambda k: self.candidates[k]['val_score'])
        return best_id

  def run_replay_and_evaluate(self, chosen_id=None):
        if chosen_id is None or chosen_id not in self.candidates:
            chosen_id = self._trigger_fallback()


        best_candidate = self.candidates[chosen_id]
        agent_code_used = best_candidate["agent_code"] 
        expected_val_score = best_candidate["val_score"]

        try:
            replay_vars = {}

            exec(agent_code_used, {}, replay_vars)

            reproduced_model = replay_vars.get('model')
            if reproduced_model is None:
                raise KeyError("В коде агента после повторного запуска не найден объект 'model'")

            replay_predictions = reproduced_model.predict(self.val_x)
            replay_val_score = calculate_accuracy(self.val_y, replay_predictions)

            if abs(replay_val_score - expected_val_score) > 1e-5:
                print("Внимание: Код не воспроизводим! Скор при повторном запуске изменился.")
            else:
                print("Replay прошел успешно! Код стабилен.")

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            return {
                "status": "rejected",
                "message": "Финальное решение отклонено: код не воспроизводится (падает при перезапуске).",
                "error": error_msg
            }

        final_predictions = reproduced_model.predict(self.private_x)
        final_private_score = calculate_accuracy(self.private_y, final_predictions)

        return {
            "status": "success",
            "chosen_candidate_id": chosen_id,
            "validation_score": replay_val_score,
            "final_private_score": final_private_score
        }


# ______________________________________________________________________


  # В КАКОМ РЕЖИМЕ ТЕСТИРУЕМ?
    def set_mode(self, mode: str):
    allowed_modes = ["single-shot", "repeated", "fixed-transitions", "flexible"]

    if mode not in allowed_modes:
        raise ValueError(f"Неизвестный режим. Выберите из: {allowed_modes}")

    self.mode = mode

    # сброс состояния
    self.attempts_made = 0
    self.current_stage = "EDA"  # (для режима Fixed)

    print(f"Среда переведена в режим: {mode.upper()}")


    # ПРИВАТНАЯ ВНУТРЕННЯЯ ПЕСОЧНИЦА (запуск кода)
    def _execute_code(self, agent_code: str) -> dict:
    try:
        # 1. Создание песочницы
        local_vars = {"train_df": self.train_data}

        # Запускаем сырой код, который прислал ИИ-агент
        exec(agent_code, {}, local_vars)

        # Пытаемся достать обученную модель из контекста
        model = local_vars.get("model")
        if model is None:
            return {"success": False, "error": "Объект 'model' не найден в вашем коде."}

        # 2. Расчет метрики на валидации
        preds = model.predict(self.val_x)

        val_score = accuracy_score(self.val_y, preds)

        # 3. Автоматическая регистрация кандидата
        attempt_id = f"attempt_{len(self.candidates) + 1}"
        self.candidates[attempt_id] = {
            "agent_code": agent_code,
            "val_score": val_score
        }

        # Возвращаем среде полный отчет об успешном запуске
        return {"success": True, "val_score": val_score, "candidate_id": attempt_id, "error": None}

    except Exception as e:
        # Если код агента упал, ловим ошибку, превращаем её в текст и возвращаем наверх
        import traceback
        return {"success": False, "val_score": None, "error": traceback.format_exc()}



    # ГЛАВНЫЙ МЕТОД. Проверяет правила -> выполняет код (_execute_code())-> дает фидбек в соответствии с режимом
    def step(self, agent_code: str, stage_action: str = None) -> dict:

        self.attempts_made += 1

        # --- РЕЖИМ 1: SINGLE-SHOT ---
        if self.mode == "single-shot":
            if self.attempts_made > 1:
                return {"status": "Rejected", "error": "В режиме Single-shot разрешена только 1 попытка."}

            # Запускаем код один раз
            res = self._execute_code(agent_code)

            # Фидбек абсолютно «слепой» — агент не знает, угадал он или нет
            return {
                "status": "Finished",
                "message": "Код успешно принят и сохранен. Промежуточный фидбек отключен организаторами."
            }

        # --- РЕЖИМ 2: REPEATED SINGLE-SHOT ---
        elif self.mode == "repeated":
            if self.attempts_made > self.max_attempts:
                return {"status": "Rejected", "error": f"Превышен лимит попыток ({self.max_attempts})."}

            res = self._execute_code(agent_code)

            # Защита данных: если код упал, мы НЕ возвращаем traceback
            if not res["success"]:
                return {
                    "status": "Execution Error",
                    "validation_score": None,
                    "message": "В коде произошла ошибка. Логи компилятора скрыты настройками режима."
                }

            # Если все ок — возвращаем СТРОГО только число
            return {
                "status": "Success",
                "validation_score": res["val_score"]
            }

        # --- РЕЖИМ 3: FIXED TRANSITIONS ---
        elif self.mode == "fixed-transitions":
            pipeline = ["EDA", "FEATURES", "TRAIN"]

            if stage_action not in pipeline:
                return {"status": "Rejected", "error": f"Неизвестный шаг. Возможные шаги: {pipeline}"}

            # Проверяем, совпадает ли вызванный шаг с тем, который ожидает среда
            if stage_action != self.current_stage:
                return {
                    "status": "Pipeline Violation",
                    "error": f"Нарушена последовательность! Сейчас вы должны выполнять этап: {self.current_stage}. "
                             f"Вызов этапа {stage_action} заблокирован."
                }

            # Выполняем код агента для текущего шага
            res = self._execute_code(agent_code)

            if not res["success"]:
                # В интерактивном режиме возвращаем ПОЛНЫЙ traceback ошибки!
                return {"status": "Runtime Error", "traceback": res["error"]}

            # Переводим конвейер на следующий этап
            current_idx = pipeline.index(self.current_stage)
            if current_idx < len(pipeline) - 1:
                self.current_stage = pipeline[current_idx + 1]

            return {
                "status": "Success",
                "validation_score": res["val_score"],
                "message": f"Этап {stage_action} пройден успешно. Следующий обязательный этап: {self.current_stage}"
            }

          # --- РЕЖИМ 4: FLEXIBLE TRANSITIONS ---
          elif self.mode == "flexible":
              res = self._execute_code(agent_code)

              # Полная свобода действий и максимальный фидбек
              if not res["success"]:
                  return {
                      "status": "Runtime Error",
                      "traceback": res["error"],
                      "hint": "Проверьте совместимость типов данных или размерность матриц."
                  }

              return {
                  "status": "Success",
                  "validation_score": res["val_score"],
                  "candidate_id": res["candidate_id"]
              }