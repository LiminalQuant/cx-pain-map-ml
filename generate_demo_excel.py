from pathlib import Path

import numpy as np
import pandas as pd


def make_demo_data(seed: int = 42, rows: int = 260) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", "2026-05-31", freq="D")
    clinics = ["Белорусская", "Красная Пресня", "Мичуринский", "Отрадное", "Марьино"]
    channels = ["Телефон", "Приложение", "Сайт", "Личный визит", "Email"]

    pain_texts = [
        "Ждал прием почти час, очередь у регистратуры, никто не предупредил о задержке",
        "Не смог записаться через приложение, ошибка оплаты и зависание личного кабинета",
        "Страховая не подтвердила ДМС, на кассе долго разбирались со счетом",
        "Администратор грубо разговаривала и не объяснила куда идти",
        "Не пришли результаты анализов, врач не оставил нормальное заключение",
        "Не нашел кабинет, нет указателей, парковка неудобная",
        "Не дозвонился в колл центр, запись отменили без предупреждения",
        "Нужна была справка, документы готовили слишком долго",
        "Врач внимательно провел прием, но ожидание и очередь испортили впечатление",
        "СМС-код не приходит, приложение не открывается, оплатить невозможно",
    ]
    neutral_texts = [
        "В целом нормально, но ожидание было долгим",
        "К врачу претензий нет, но запись через сайт неудобная",
        "Не хватило информации по документам и дальнейшим действиям",
        "Все решили, но пришлось несколько раз звонить",
    ]
    good_texts = ["Все хорошо, врач помог", "Быстро приняли, спасибо", "Нормально", "Отличный прием"]

    nps_rows = []
    for _ in range(rows):
        score = int(rng.choice([2, 4, 5, 6, 7, 8, 9, 10], p=[0.05, 0.08, 0.09, 0.12, 0.16, 0.16, 0.18, 0.16]))
        if score <= 6:
            segment = "Критик"
            comment = rng.choice(pain_texts)
        elif score <= 8:
            segment = "Нейтрал"
            comment = rng.choice(neutral_texts)
        else:
            segment = "Промоутер"
            comment = rng.choice(good_texts)

        nps_rows.append({
            "Дата талона": rng.choice(dates),
            "Название клиники": rng.choice(clinics),
            "Тип респондента": segment,
            "Оценка": score,
            "Вопрос": rng.choice(["Оцените визит", "Оцените запись", "Оцените приложение", "Оцените ожидание"]),
            "Комментарий": comment,
            "Опция ответа": rng.choice(["", "Долго ждать", "Неудобная запись", "Проблема с оплатой", "Персонал"]),
        })

    crm_rows = []
    statuses = ["Закрыто", "В работе", "Повторно", "Эскалация"]
    categories = ["Жалоба", "Вопрос", "Претензия", "Благодарность", "Техническая проблема"]
    units = ["Регистратура", "Контакт-центр", "ДМС", "IT", "Врачи", "Документы"]
    for _ in range(rows):
        crm_rows.append({
            "Дата обращения": rng.choice(dates),
            "Клиника": rng.choice(clinics),
            "Канал": rng.choice(channels),
            "Статус": rng.choice(statuses, p=[0.55, 0.25, 0.12, 0.08]),
            "Категория": rng.choice(categories, p=[0.38, 0.22, 0.18, 0.07, 0.15]),
            "Ответственный блок": rng.choice(units),
            "Текст обращения": rng.choice(pain_texts + neutral_texts),
        })

    return pd.DataFrame(nps_rows), pd.DataFrame(crm_rows)


if __name__ == "__main__":
    out = Path("demo_nps_crm.xlsx")
    nps, crm = make_demo_data()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        nps.to_excel(writer, sheet_name="nps_demo", index=False)
        crm.to_excel(writer, sheet_name="crm_demo", index=False)
    print(out.resolve())
