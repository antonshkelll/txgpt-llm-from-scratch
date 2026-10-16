"""
Синтетические описания банковских операций (как в выписке) с категориями трат.

Зачем: у оплат по СБП/QR и переводов самозанятым часто нет MCC-кода, поэтому банк
вынужден определять категорию по тексту описания. Реальные выписки — банковская тайна,
поэтому генерируем похожие строки: разные каналы (POS-терминал латиницей/кириллицей,
онлайн, СБП, переводы), шум (регистр, опечатки, обрезка, номера терминалов).

Ключевое для честной оценки: часть мерчантов каждой категории полностью убрана из
РАЗМЕТКИ (split 'unseen') — модель должна обобщать, а не запоминать бренды.
При этом в неразмеченном потоке для предобучения есть все мерчанты (как у банка:
описаний миллионы, а размечена малая часть), но метки оттуда не используются.
"""
from __future__ import annotations

import random

CATEGORIES = {
    "Супермаркеты": ["Пятёрочка", "Перекрёсток", "Магнит", "Лента", "Ашан", "ВкусВилл", "Дикси", "Азбука вкуса",
                     "Spar", "Окей", "Глобус", "Верный", "Метро", "Билла", "Монетка", "Светофор", "Самокат", "Чижик"],
    "Кафе и рестораны": ["Шоколадница", "Кофе Хауз", "Теремок", "Му-Му", "Тануки", "Якитория", "Cofix", "Surf Coffee",
                         "Даблби", "Стардогс", "Хинкальная", "Додо Пицца", "Чайхона №1", "Кофемания", "Coffee Like",
                         "One Price Coffee"],
    "Фастфуд и доставка": ["Вкусно и точка", "Ростикс", "Бургер Кинг", "Яндекс Еда", "Delivery Club", "Subway",
                           "Шаурма Хаус", "Кухня на районе", "Папа Джонс", "Самокат Еда", "Сбермаркет Кухня"],
    "Такси": ["Яндекс Go", "Ситимобил", "Uber", "Максим", "Везёт", "Таксовичкоф", "Drivee", "inDrive"],
    "Транспорт": ["Метрополитен", "Мосгортранс", "Тройка", "РЖД", "Аэрофлот", "Победа", "S7", "Аэроэкспресс",
                  "Whoosh", "Юрент", "Делимобиль", "Ситидрайв", "Туту"],
    "АЗС": ["Лукойл", "Роснефть", "Газпромнефть", "Татнефть", "Shell", "Teboil", "Башнефть", "ТНК", "Нефтьмагистраль"],
    "Маркетплейсы": ["Ozon", "Wildberries", "Яндекс Маркет", "Мегамаркет", "AliExpress", "Lamoda", "Авито",
                     "Детский мир", "KazanExpress", "Joom"],
    "Одежда и обувь": ["Zarina", "Befree", "Gloria Jeans", "Спортмастер", "Love Republic", "Lime", "Ostin",
                       "Ralf Ringer", "Kari", "Snow Queen", "Familia", "Твое", "Rendez-Vous", "Melon Fashion"],
    "Аптеки и здоровье": ["Аптека Ригла", "Горздрав", "Апрель", "Аптека Вита", "36,6", "Столички", "Еаптека",
                          "Инвитро", "Гемотест", "СМ-Клиника", "Медси", "Доктор Рядом", "Планета Здоровья"],
    "Связь и интернет": ["МТС", "Билайн", "Мегафон", "Tele2", "Ростелеком", "Йота", "Дом.ру", "Тинькофф Мобайл",
                         "СберМобайл", "Мотив"],
    "ЖКХ": ["Мосэнергосбыт", "ЕИРЦ", "МОЭК", "Мосводоканал", "Петроэлектросбыт", "Газпром межрегионгаз",
            "ЖКУ Москвы", "Водоканал", "УК Комфорт", "Энергосбыт"],
    "Подписки и развлечения": ["Яндекс Плюс", "Кинопоиск", "Okko", "Иви", "Wink", "VK Музыка", "Литрес",
                               "Звук", "Синема Парк", "Каро", "Формула Кино", "Steam", "PlayStation Store",
                               "Яндекс Афиша"],
    "Образование": ["Skillbox", "Нетология", "Яндекс Практикум", "GeekBrains", "Фоксфорд", "Skyeng", "Учи.ру",
                    "Stepik", "Тетрика", "Умскул", "Coursera"],
    "Красота": ["Летуаль", "Рив Гош", "Иль де Ботэ", "Подружка", "Золотое Яблоко Бьюти", "Барбершоп Topgun",
                "Chop-Chop", "Студия маникюра", "Салон Персона", "OldBoy"],
}

# «Общие» слова, по которым человек угадал бы категорию у незнакомого мерчанта
GENERIC = {
    "Супермаркеты": ["ПРОДУКТЫ", "МИНИМАРКЕТ", "УНИВЕРСАМ", "ГАСТРОНОМ", "SUPERMARKET", "PRODUKTY"],
    "Кафе и рестораны": ["КАФЕ", "РЕСТОРАН", "КОФЕЙНЯ", "COFFEE", "CAFE", "БАР", "БИСТРО"],
    "Фастфуд и доставка": ["ШАУРМА", "БУРГЕРНАЯ", "ПИЦЦА", "ДОСТАВКА ЕДЫ", "FOOD", "PIZZA"],
    "Такси": ["ТАКСИ", "TAXI", "ТРАНСФЕР"],
    "Транспорт": ["ПАРКОВКА", "БИЛЕТ", "КАРШЕРИНГ", "TRANSPORT", "АВТОБУС", "ЭЛЕКТРИЧКА"],
    "АЗС": ["АЗС", "AZS", "ТОПЛИВО", "FUEL", "GAS STATION"],
    "Маркетплейсы": ["МАРКЕТПЛЕЙС", "MARKETPLACE", "ИНТЕРНЕТ-МАГАЗИН", "SHOP ONLINE"],
    "Одежда и обувь": ["ОДЕЖДА", "ОБУВЬ", "FASHION", "SHOES", "БУТИК"],
    "Аптеки и здоровье": ["АПТЕКА", "APTEKA", "КЛИНИКА", "СТОМАТОЛОГИЯ", "ЛАБОРАТОРИЯ", "PHARMACY"],
    "Связь и интернет": ["СВЯЗЬ", "ИНТЕРНЕТ", "МОБИЛЬНАЯ СВЯЗЬ", "TELECOM", "ПРОВАЙДЕР"],
    "ЖКХ": ["ЖКУ", "ЭНЕРГОСБЫТ", "ВОДОКАНАЛ", "УПРАВЛЯЮЩАЯ КОМПАНИЯ", "КАПРЕМОНТ"],
    "Подписки и развлечения": ["КИНОТЕАТР", "ПОДПИСКА", "SUBSCRIPTION", "GAMES", "ТЕАТР", "КОНЦЕРТ"],
    "Образование": ["ШКОЛА", "КУРСЫ", "ОБУЧЕНИЕ", "РЕПЕТИТОР", "ACADEMY", "УНИВЕРСИТЕТ"],
    "Красота": ["САЛОН КРАСОТЫ", "БАРБЕРШОП", "МАНИКЮР", "КОСМЕТИКА", "BEAUTY", "СПА"],
}
TRANSFER = "Переводы"
ALL_CATEGORIES = list(CATEGORIES) + [TRANSFER]

CITIES = ["МОСКВА", "САНКТ-ПЕТЕРБУРГ", "КАЗАНЬ", "ЕКАТЕРИНБУРГ", "НОВОСИБИРСК", "НИЖНИЙ НОВГОРОД", "САМАРА",
          "КРАСНОДАР", "ТВЕРЬ", "ТУЛА", "ВОРОНЕЖ", "ПЕРМЬ"]
NAMES = ["Анна", "Иван", "Мария", "Дмитрий", "Елена", "Алексей", "Ольга", "Сергей", "Наталья", "Артём", "Ксения"]
SURNAMES = ["Иванов", "Смирнов", "Кузнецов", "Попов", "Васильев", "Петров", "Соколов", "Михайлов", "Новиков",
            "Фёдоров", "Морозов", "Волков", "Лебедев"]
SYLL = ["ра", "мо", "ли", "ка", "то", "ве", "ни", "сол", "дар", "мир", "лан", "тер", "бо", "ста", "вит", "ро"]

_TR = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
               ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p", "r", "s", "t",
                "u", "f", "kh", "ts", "ch", "sh", "sch", "", "y", "", "e", "yu", "ya"]))


def translit(s: str) -> str:
    return "".join(_TR.get(ch, ch) if ch.islower() else _TR.get(ch.lower(), ch).upper() for ch in s)


def noise(s: str, r: random.Random) -> str:
    if r.random() < 0.15 and len(s) > 6:                       # опечатка
        i = r.randrange(1, len(s) - 1)
        s = s[:i] + s[i + 1:] if r.random() < 0.5 else s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if r.random() < 0.1 and len(s) > 12:
        s = s[: r.randint(max(8, len(s) // 2), len(s))]          # обрезка банковской системой
    if r.random() < 0.2:
        s = s.lower()
    return " ".join(s.split())


def fake_name(r: random.Random) -> str:
    return "".join(r.choice(SYLL) for _ in range(r.randint(2, 3))).capitalize()


def describe(merchant: str, cat: str, r: random.Random) -> str:
    """Одна строка выписки для мерчанта — в одном из форматов разных каналов оплаты."""
    city = r.choice(CITIES)
    if r.random() < 0.25:                                        # как в реальных POS: «АПТЕКА РИГЛА», «КЛИНИКА МЕДСИ»
        merchant = f"{r.choice(GENERIC[cat]).split()[0].capitalize()} {merchant}"
    tid = str(r.randint(10, 99999))
    fmt = r.random()
    if fmt < 0.35:                                               # POS-терминал, латиница
        s = f"{translit(merchant).upper()} {tid} {translit(city.lower()).upper()} RUS"
    elif fmt < 0.6:                                              # POS, кириллица
        s = f"{merchant.upper()} {tid} {city}"
    elif fmt < 0.75:                                             # онлайн
        s = r.choice([f"{translit(merchant).upper().replace(' ', '')}*{tid}",
                      f"WWW.{translit(merchant).upper().replace(' ', '')}.RU",
                      f"{merchant} онлайн"])
    else:                                                        # СБП / QR
        org = r.choice(["ООО", "АО", "ИП"])
        s = f"Оплата СБП {org} «{merchant}»" if r.random() < 0.5 else f"QR {merchant.upper()} {city}"
    return noise(s, r)


def describe_generic(cat: str, r: random.Random) -> str:
    """Незнакомый мерчант, у которого в названии есть «подсказка» категории."""
    word = r.choice(GENERIC[cat])
    name = fake_name(r)
    if r.random() < 0.3:
        return noise(f"Оплата СБП ИП {r.choice(SURNAMES)} {r.choice(NAMES)[0]}.{r.choice(NAMES)[0]}. ({word.lower()})", r)
    w = translit(word) if (word.isascii() is False and r.random() < 0.3) else word
    return noise(f"{w} {name.upper()} {r.randint(1, 999)} {r.choice(CITIES)}", r)


def describe_transfer(r: random.Random) -> str:
    n, s = r.choice(NAMES), r.choice(SURNAMES)
    return noise(r.choice([
        f"Перевод по номеру телефона +7 9** ***-{r.randint(10, 99)}-{r.randint(10, 99)} {n} {s[0]}.",
        f"Перевод СБП {n} {s[0]}.",
        f"Внутренний перевод на карту ****{r.randint(1000, 9999)}",
        f"Перевод клиенту {r.choice(['Т-Банк', 'Сбербанк', 'Альфа-Банк', 'ВТБ'])} {n} {s[0]}.",
    ]), r)


def build_dataset(n: int, seed: int = 0, unseen_share: float = 0.25):
    """Возвращает (train, test_seen, test_unseen, pretrain_texts) как списки (текст, категория)."""
    r = random.Random(seed)
    seen, unseen = {}, {}
    for cat, ms in CATEGORIES.items():
        ms = ms[:]
        r.shuffle(ms)
        k = max(2, round(len(ms) * unseen_share))
        unseen[cat], seen[cat] = ms[:k], ms[k:]

    def sample(pool, m):
        out = []
        for _ in range(m):
            u = r.random()
            if u < 0.08:
                out.append((describe_transfer(r), TRANSFER))
                continue
            cat = r.choice(list(CATEGORIES))
            if u < 0.30:
                out.append((describe_generic(cat, r), cat))
            else:
                out.append((describe(r.choice(pool[cat]), cat, r), cat))
        return out

    train = sample(seen, n)
    test_seen = sample(seen, n // 10)
    test_unseen = sample(unseen, n // 10)
    every = {c: seen[c] + unseen[c] for c in CATEGORIES}
    pretrain = [t for t, _ in sample(every, n * 4)]   # неразмеченный поток: все мерчанты, БЕЗ меток
    return train, test_seen, test_unseen, pretrain


if __name__ == "__main__":
    tr, ts, tu, pt = build_dataset(2000)
    r = random.Random(1)
    for t, c in r.sample(tr, 12):
        print(f"{c:24s} | {t}")
    print("--- unseen merchants:")
    for t, c in r.sample(tu, 6):
        print(f"{c:24s} | {t}")
