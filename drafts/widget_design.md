Промпт для Cursor: первый экран чат-виджета

Нужно реализовать первый приветственный экран для уже описанного медицинского чат-виджета. Он должен быть визуально и технически синхронизирован с экраном диалога, который уже описан в предыдущем ТЗ.

Виджет должен иметь два состояния:

Welcome state — первый экран приветствия.
Dialog state — экран переписки с сообщениями бота и пользователя.

Сейчас нужно сверстать именно Welcome state, но архитектурно предусмотреть плавный переход в Dialog state после клика по кнопке быстрого действия или после ввода/отправки сообщения.

1. Базовые размеры и каркас

Использовать те же размеры, что и для экрана диалога.

--widget-width: 420px;
--widget-height: 720px;

--frame-padding: 12px;
--radius-widget: 30px;
--radius-surface: 22px;

--space-x: 28px;

Внешний размер:

.chat-widget {
  width: 420px;
  height: min(720px, calc(100vh - 48px));
}

Для ноутбуков 1366×768:

@media (max-height: 800px) {
  .chat-widget {
    width: 400px;
    height: min(680px, calc(100vh - 40px));
  }
}
2. Архитектура слоёв

Сохраняем ту же структуру, что в экране диалога:

<div class="chat-widget">
  <div class="chat-widget__frame">
    <div class="chat-widget__surface">
      <header class="chat-header">...</header>

      <main class="chat-welcome">
        ...
      </main>

      <form class="chat-input-area">
        ...
      </form>
    </div>
  </div>
</div>
Важно

chat-widget__frame — это внешняя градиентная подложка.

chat-widget__surface — внутренняя белая полупрозрачная рабочая область.

Идея такая:

градиентные пятна находятся на внешнем каркасе;
внутренняя рабочая область белая, но слегка прозрачная;
из-за этого градиенты мягко просвечивают;
между каркасом и рабочей областью есть отступ 12px;
получается эффект лёгкой светящейся рамки.
3. Цвета

Использовать те же CSS-переменные, что и на экране диалога.

:root {
  --color-text: #10233F;
  --color-text-soft: #738197;
  --color-text-muted: #9AA7B8;

  --color-primary: #08B6C4;
  --color-primary-dark: #0798A5;
  --color-primary-light: #7FEAE1;

  --color-line: rgba(16, 35, 63, 0.10);
  --color-border: rgba(8, 182, 196, 0.85);

  --gradient-primary: linear-gradient(135deg, #50E4C7 0%, #16C3D6 52%, #2499E6 100%);
  --gradient-online: linear-gradient(135deg, #8AF447 0%, #32D33D 48%, #10B83A 100%);
}

Каркас:

.chat-widget__frame {
  width: 100%;
  height: 100%;
  border-radius: 30px;
  padding: 12px;
  position: relative;
  overflow: hidden;
  background:
    radial-gradient(circle at 82% 5%, rgba(105, 235, 220, 0.42), transparent 32%),
    radial-gradient(circle at 0% 62%, rgba(94, 226, 209, 0.28), transparent 34%),
    radial-gradient(circle at 100% 88%, rgba(72, 175, 238, 0.24), transparent 34%),
    linear-gradient(145deg, #ffffff 0%, #f7fcff 46%, #eefaff 100%);
  box-shadow:
    0 24px 60px rgba(15, 80, 110, 0.16),
    0 8px 24px rgba(12, 90, 120, 0.08);
}

Рабочая область:

.chat-widget__surface {
  width: 100%;
  height: 100%;
  border-radius: 22px;
  overflow: hidden;
  background: rgba(255, 255, 255, 0.78);
  backdrop-filter: blur(22px);
  -webkit-backdrop-filter: blur(22px);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.8),
    0 10px 30px rgba(20, 95, 120, 0.08);
  display: flex;
  flex-direction: column;
}
4. Header

Header полностью совпадает с экраном диалога.

В шапке:

аватар Надежды;
зелёная градиентная точка онлайн в правом нижнем углу аватара;
имя: Надежда;
подпись: онлайн-консультант;
справа деликатный крестик закрытия;
без стрелки назад;
без меню-троеточия.
<header class="chat-header">
  <div class="chat-header__avatar">
    <img src="/avatar.jpg" alt="Надежда">
    <span class="chat-header__online"></span>
  </div>

  <div class="chat-header__text">
    <div class="chat-header__name">Надежда</div>
    <div class="chat-header__status">онлайн-консультант</div>
  </div>

  <button class="chat-close" type="button" aria-label="Закрыть чат">
    <!-- close svg -->
  </button>
</header>

CSS:

.chat-header {
  position: relative;
  display: flex;
  align-items: center;
  gap: 18px;
  padding: 26px 28px 18px;
  flex-shrink: 0;
}

.chat-header__avatar {
  width: 64px;
  height: 64px;
  border-radius: 50%;
  position: relative;
  flex-shrink: 0;
  box-shadow: 0 8px 18px rgba(20, 110, 130, 0.14);
}

.chat-header__avatar img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: inherit;
}

.chat-header__online {
  width: 18px;
  height: 18px;
  position: absolute;
  right: 1px;
  bottom: 3px;
  border-radius: 50%;
  background: var(--gradient-online);
  box-shadow:
    0 0 0 3px rgba(255, 255, 255, 0.95),
    0 4px 10px rgba(34, 205, 62, 0.35);
}

.chat-header__name {
  font-size: 28px;
  line-height: 1.1;
  font-weight: 700;
  letter-spacing: -0.03em;
  color: var(--color-text);
}

.chat-header__status {
  margin-top: 5px;
  font-size: 15px;
  line-height: 1.35;
  font-weight: 500;
  color: var(--color-text-soft);
}

Крестик:

.chat-close {
  position: absolute;
  top: 28px;
  right: 28px;
  width: 32px;
  height: 32px;
  border: 0;
  background: transparent;
  color: #748198;
  cursor: pointer;
  display: grid;
  place-items: center;
  border-radius: 50%;
  transition: background-color 0.18s ease, color 0.18s ease, transform 0.18s ease;
}

.chat-close:hover {
  background: rgba(16, 35, 63, 0.055);
  color: #31425B;
}

.chat-close:active {
  transform: scale(0.94);
}
5. Welcome body

Главная область первого экрана.

Она должна занимать всё пространство между header и input.

<main class="chat-welcome">
  <section class="welcome-card">
    <div class="welcome-logo-wrap">
      <div class="welcome-logo">
        <!-- clinic logo svg -->
      </div>

      <span class="welcome-spark welcome-spark--one"></span>
      <span class="welcome-spark welcome-spark--two"></span>
      <span class="welcome-spark welcome-spark--three"></span>
    </div>

    <div class="welcome-text">
      <strong>Здравствуйте!</strong>
      <span>Я помогу сориентироваться</span>
      <span>по стоимости, этапам лечения</span>
      <span>
        и записи на консультацию.<i class="stream-cursor"></i>
      </span>
    </div>

    <div class="welcome-wave" aria-hidden="true"></div>
  </section>

  <div class="welcome-actions">
    <button class="welcome-action" type="button">Узнать стоимость</button>
    <button class="welcome-action" type="button">Как проходит лечение?</button>
    <button class="welcome-action" type="button">Записаться на консультацию</button>
  </div>
</main>
Общая логика welcome body
приветствие в большой белой карточке с тенью;
карточка чуть выше центра, не слишком высокая;
под карточкой три кнопки в колонку;
карточка и три кнопки должны быть одинаковой ширины;
эта ширина должна визуально соответствовать ширине нижнего input-area;
кнопки без иконок;
текст в кнопках строго по центру;
общий стиль — medical premium / soft glass;
никакой лишней второй навигации, никаких дополнительных меню.

CSS:

.chat-welcome {
  flex: 1;
  min-height: 0;
  padding: 8px 28px 18px;
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
}
6. Ширина контентной зоны

Для welcome-card, welcome-actions и input нужно использовать единую визуальную сетку.

Основная ширина контента внутри рабочей области:

--content-width: 100%;

Поскольку chat-welcome имеет padding-inline: 28px, все блоки внутри него могут быть width: 100%.

.welcome-card,
.welcome-actions,
.welcome-action {
  width: 100%;
}

Важно: приветственная карточка и кнопки должны быть по ширине одинаковыми.

Нижний input-area остаётся с теми же боковыми отступами 28px, но внутри там есть textarea + send button. Визуально welcome-card и кнопки должны совпадать по левой границе с textarea, а по правой — идти до правой границы контентной зоны. Если нужно абсолютное совпадение с полной строкой input + send, можно сделать input-area такой же ширины, а кнопку отправки оставить справа внутри этой строки. Но в текущем дизайне send button отдельный, поэтому достаточно выровнять welcome-card и кнопки по основной контентной сетке.

7. Приветственная карточка

Карточка:

белая / слегка полупрозрачная;
мягкая тень;
радиус 22–24 px;
находится чуть выше центра;
высота компактнее предыдущего варианта;
внутри сверху логотип клиники;
вокруг логотипа 2–3 маленькие четырёхугольные звёздочки;
крупный центрированный текст;
в конце текста бирюзовый мигающий cursor.
.welcome-card {
  position: relative;
  margin-top: 34px;
  min-height: 285px;
  padding: 78px 24px 34px;
  border-radius: 24px;
  background: rgba(255, 255, 255, 0.72);
  border: 1px solid rgba(255, 255, 255, 0.68);
  box-shadow:
    0 22px 44px rgba(18, 82, 100, 0.12),
    0 6px 16px rgba(18, 82, 100, 0.055),
    inset 0 1px 0 rgba(255, 255, 255, 0.82);
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
  overflow: visible;
}

Для экранов 1366×768 можно немного уменьшить:

@media (max-height: 800px) {
  .welcome-card {
    margin-top: 20px;
    min-height: 250px;
    padding-top: 68px;
    padding-bottom: 28px;
  }
}
8. Логотип клиники

Пока точного логотипа нет, нужно сделать placeholder-компонент ClinicLogo. Его потом можно заменить на реальный SVG клиники.

Логотип должен быть:

в круглом бейдже;
расположен по центру сверху карточки;
частично “висит” над карточкой;
цвет — бирюза;
стиль — минималистичный медицинский знак, не мультяшный.
<div class="welcome-logo">
  <svg viewBox="0 0 64 64" fill="none" aria-hidden="true">
    <path d="M28 13H36V28H51V36H36V51H28V36H13V28H28V13Z"
          stroke="currentColor"
          stroke-width="4"
          stroke-linejoin="round"/>
    <path d="M36 43C44 35 52 35 54 36C52 46 44 51 36 50C36 47 36 45 36 43Z"
          stroke="currentColor"
          stroke-width="3.4"
          stroke-linecap="round"
          stroke-linejoin="round"/>
    <path d="M37 48C41 44 45 41 50 39"
          stroke="currentColor"
          stroke-width="2.8"
          stroke-linecap="round"/>
  </svg>
</div>

CSS:

.welcome-logo-wrap {
  position: absolute;
  top: -48px;
  left: 50%;
  width: 116px;
  height: 116px;
  transform: translateX(-50%);
  display: grid;
  place-items: center;
  pointer-events: none;
}

.welcome-logo {
  width: 86px;
  height: 86px;
  border-radius: 50%;
  display: grid;
  place-items: center;
  color: var(--color-primary);
  background:
    radial-gradient(circle at 45% 30%, rgba(255,255,255,0.95), rgba(255,255,255,0.74) 62%, rgba(236, 255, 253, 0.68) 100%);
  border: 1px solid rgba(8, 182, 196, 0.65);
  box-shadow:
    0 0 0 10px rgba(8, 182, 196, 0.055),
    0 12px 30px rgba(8, 182, 196, 0.16),
    inset 0 1px 0 rgba(255,255,255,0.9);
}

.welcome-logo svg {
  width: 46px;
  height: 46px;
}

Когда появится реальный логотип клиники, заменить SVG внутри .welcome-logo.

9. Звёздочки вокруг логотипа

Нужны 2–3 четырёхугольные звёздочки разного размера.

Не использовать эмодзи. Сделать CSS-фигуры или SVG.

Вариант на CSS:

<span class="welcome-spark welcome-spark--one"></span>
<span class="welcome-spark welcome-spark--two"></span>
<span class="welcome-spark welcome-spark--three"></span>
.welcome-spark {
  position: absolute;
  display: block;
  width: 14px;
  height: 14px;
  color: var(--color-primary);
  opacity: 0.72;
}

.welcome-spark::before,
.welcome-spark::after {
  content: "";
  position: absolute;
  inset: 0;
  margin: auto;
  background: currentColor;
  border-radius: 999px;
}

.welcome-spark::before {
  width: 2px;
  height: 100%;
}

.welcome-spark::after {
  width: 100%;
  height: 2px;
}

.welcome-spark {
  transform: rotate(45deg);
  filter: drop-shadow(0 4px 8px rgba(8, 182, 196, 0.16));
}

.welcome-spark--one {
  left: -56px;
  top: 20px;
  width: 15px;
  height: 15px;
}

.welcome-spark--two {
  right: -52px;
  top: 18px;
  width: 17px;
  height: 17px;
  opacity: 0.64;
}

.welcome-spark--three {
  right: -28px;
  top: 58px;
  width: 10px;
  height: 10px;
  opacity: 0.52;
}

Можно добавить лёгкое “дыхание”:

.welcome-spark {
  animation: sparkPulse 2.8s ease-in-out infinite;
}

.welcome-spark--two {
  animation-delay: 0.45s;
}

.welcome-spark--three {
  animation-delay: 0.9s;
}

@keyframes sparkPulse {
  0%, 100% {
    opacity: 0.45;
    transform: rotate(45deg) scale(0.9);
  }
  50% {
    opacity: 0.8;
    transform: rotate(45deg) scale(1.05);
  }
}
10. Текст приветствия

Текст должен быть крупнее, чем обычный текст в сообщениях.

Центрированный.

Состояние должно выглядеть как streaming / typing.

Текст:

Здравствуйте!
Я помогу сориентироваться
по стоимости, этапам лечения
и записи на консультацию.
.welcome-text {
  position: relative;
  z-index: 2;
  text-align: center;
  color: var(--color-text);
  font-size: 24px;
  line-height: 1.48;
  font-weight: 650;
  letter-spacing: -0.025em;
}

.welcome-text strong {
  display: block;
  margin-bottom: 8px;
  font-size: 26px;
  line-height: 1.15;
  font-weight: 750;
  letter-spacing: -0.035em;
}

.welcome-text span {
  display: block;
}

Для компактных экранов:

@media (max-height: 800px) {
  .welcome-text {
    font-size: 21px;
    line-height: 1.43;
  }

  .welcome-text strong {
    font-size: 24px;
    margin-bottom: 6px;
  }
}
11. Бирюзовая мигающая черта стриминга

В конце последней строки текста должна быть тонкая бирюзовая вертикальная черта.

<span>
  и записи на консультацию.<i class="stream-cursor"></i>
</span>

CSS:

.stream-cursor {
  display: inline-block;
  width: 2px;
  height: 1.1em;
  margin-left: 8px;
  vertical-align: -0.12em;
  border-radius: 999px;
  background: var(--color-primary);
  animation: cursorBlink 0.9s steps(2, start) infinite;
}

@keyframes cursorBlink {
  0%, 45% {
    opacity: 1;
  }
  46%, 100% {
    opacity: 0;
  }
}

Если будет настоящий streaming текста, курсор показывать только пока текст печатается. После завершения можно скрывать через класс:

.welcome-card.is-finished .stream-cursor {
  display: none;
}
12. Декоративные волны внизу welcome-card

В карточке внизу сделать очень деликатные бирюзовые волны, как на референсе.

Можно через псевдоэлемент:

.welcome-wave {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  height: 58px;
  border-radius: 0 0 24px 24px;
  overflow: hidden;
  pointer-events: none;
  opacity: 0.55;
}

.welcome-wave::before,
.welcome-wave::after {
  content: "";
  position: absolute;
  left: -10%;
  right: -10%;
  height: 44px;
  border-top: 1px solid rgba(8, 182, 196, 0.16);
  border-radius: 50%;
}

.welcome-wave::before {
  bottom: -20px;
  transform: rotate(-2deg);
}

.welcome-wave::after {
  bottom: -8px;
  transform: rotate(3deg);
  opacity: 0.6;
}

Волны должны быть едва заметными. Не должны спорить с текстом.

13. Кнопки быстрого действия

Три кнопки:

Узнать стоимость
Как проходит лечение?
Записаться на консультацию

Требования:

все три в одну колонку;
без иконок;
одинаковая ширина с welcome-card;
текст строго по центру;
стиль outline / glass;
белый или слегка прозрачный фон;
бирюзовая обводка;
бирюзовый текст;
радиус как в диалоговом экране;
визуально явно выглядят как кнопки.
<div class="welcome-actions">
  <button class="welcome-action" type="button" data-action="price">
    Узнать стоимость
  </button>

  <button class="welcome-action" type="button" data-action="process">
    Как проходит лечение?
  </button>

  <button class="welcome-action" type="button" data-action="consultation">
    Записаться на консультацию
  </button>
</div>

CSS:

.welcome-actions {
  width: 100%;
  margin-top: 22px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.welcome-action {
  width: 100%;
  height: 52px;
  border-radius: 14px;
  border: 1px solid var(--color-border);
  background: rgba(255, 255, 255, 0.62);
  color: var(--color-primary);
  cursor: pointer;

  display: flex;
  align-items: center;
  justify-content: center;

  font-family: inherit;
  font-size: 15px;
  line-height: 1;
  font-weight: 500;
  letter-spacing: -0.01em;
  text-align: center;

  box-shadow:
    0 8px 18px rgba(20, 130, 150, 0.055),
    inset 0 1px 0 rgba(255,255,255,0.72);

  transition:
    transform 0.16s ease,
    box-shadow 0.16s ease,
    background-color 0.16s ease,
    border-color 0.16s ease,
    color 0.16s ease;
}

.welcome-action:hover {
  transform: translateY(-1px);
  background: rgba(255, 255, 255, 0.78);
  border-color: rgba(8, 182, 196, 1);
  color: var(--color-primary-dark);
  box-shadow:
    0 12px 24px rgba(20, 130, 150, 0.09),
    inset 0 1px 0 rgba(255,255,255,0.8);
}

.welcome-action:active {
  transform: translateY(0);
}

Для компактной высоты:

@media (max-height: 800px) {
  .welcome-actions {
    margin-top: 16px;
    gap: 8px;
  }

  .welcome-action {
    height: 48px;
    font-size: 14px;
  }
}
14. Input area

Input area полностью синхронизировать с экраном диалога.

Кнопка отправки находится снаружи textarea, справа.

При многострочном вводе textarea растёт вверх, а кнопка отправки остаётся снизу справа.

<form class="chat-input-area">
  <textarea
    class="chat-input"
    rows="1"
    placeholder="Введите сообщение"
    aria-label="Введите сообщение"></textarea>

  <button class="chat-send" type="submit" aria-label="Отправить сообщение">
    <!-- send svg -->
  </button>
</form>

CSS:

.chat-input-area {
  flex-shrink: 0;
  padding: 0 28px 26px;
  display: flex;
  align-items: flex-end;
  gap: 12px;
}

.chat-input {
  flex: 1;
  min-height: 48px;
  max-height: 112px;
  padding: 14px 18px;
  border-radius: 16px;
  border: 1px solid rgba(16, 35, 63, 0.08);
  background: rgba(255, 255, 255, 0.82);
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.9),
    0 8px 18px rgba(20, 80, 100, 0.07);

  color: var(--color-text);
  font-size: 14px;
  line-height: 1.45;
  font-family: inherit;
  resize: none;
  outline: none;
  overflow-y: hidden;

  transition:
    border-color 0.16s ease,
    box-shadow 0.16s ease,
    background-color 0.16s ease;
}

.chat-input::placeholder {
  color: #A4AFBF;
}

.chat-input:focus {
  border-color: rgba(8, 182, 196, 0.45);
  box-shadow:
    0 0 0 3px rgba(8, 182, 196, 0.09),
    0 8px 18px rgba(20, 80, 100, 0.07);
}

.chat-send {
  width: 52px;
  height: 52px;
  flex: 0 0 52px;
  border-radius: 50%;
  border: 0;
  background: linear-gradient(135deg, #23D6C9 0%, #11B8D0 52%, #1CA4E5 100%);
  color: #ffffff;
  cursor: pointer;

  display: grid;
  place-items: center;

  box-shadow:
    0 12px 24px rgba(14, 172, 205, 0.28),
    0 4px 10px rgba(15, 130, 170, 0.14);

  transition:
    transform 0.16s ease,
    opacity 0.16s ease,
    box-shadow 0.16s ease;
}

.chat-send:hover {
  transform: translateY(-1px);
}

.chat-send:active {
  transform: translateY(0) scale(0.96);
}

.chat-send:disabled {
  opacity: 0.42;
  cursor: default;
  transform: none;
  box-shadow: none;
}

Send SVG:

15. Поведение первого экрана
Welcome state

При первом открытии показываем:

header;
welcome-card;
три кнопки;
input-area.
Переход в Dialog state

После одного из действий:

пользователь нажал кнопку;
пользователь ввёл сообщение и отправил;
пользователь кликнул по input и отправил первый вопрос;

нужно плавно скрыть welcome-content и перейти в режим диалога.

Логика:

function startDialog(initialMessage) {
  setWelcomeVisible(false);
  setDialogVisible(true);

  if (initialMessage) {
    addUserMessage(initialMessage);
    startBotStreamingResponse(initialMessage);
  }
}
Анимация исчезновения welcome

Welcome не должен резко пропадать. Он должен мягко “затухать” и немного уходить вверх.

CSS:

.chat-welcome {
  transition:
    opacity 0.24s ease,
    transform 0.24s ease,
    filter 0.24s ease;
}

.chat-welcome.is-leaving {
  opacity: 0;
  transform: translateY(-8px);
  filter: blur(3px);
  pointer-events: none;
}

После завершения анимации можно размонтировать welcome и показать chat-body.

16. Анимация появления welcome

При открытии:

виджет появляется мягко;
welcome-card появляется с лёгким scale/translate;
кнопки появляются последовательно;
зелёная онлайн-точка может сделать один мягкий pulse;
курсор в приветствии мигает.
.chat-widget {
  animation: widgetIn 0.28s cubic-bezier(.22, .9, .3, 1) both;
}

.welcome-card {
  animation: welcomeCardIn 0.34s cubic-bezier(.22, .9, .3, 1) 0.06s both;
}

.welcome-action {
  animation: welcomeActionIn 0.28s cubic-bezier(.22, .9, .3, 1) both;
}

.welcome-action:nth-child(1) {
  animation-delay: 0.18s;
}

.welcome-action:nth-child(2) {
  animation-delay: 0.26s;
}

.welcome-action:nth-child(3) {
  animation-delay: 0.34s;
}

@keyframes widgetIn {
  from {
    opacity: 0;
    transform: translateY(16px) scale(0.98);
  }
  to {
    opacity: 1;
    transform: translateY(0) scale(1);
  }
}

@keyframes welcomeCardIn {
  from {
    opacity: 0;
    transform: translateY(10px) scale(0.985);
  }
  to {
    opacity: 1;
    transform: translateY(0) scale(1);
  }
}

@keyframes welcomeActionIn {
  from {
    opacity: 0;
    transform: translateY(8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

Отключение анимаций для reduced motion:

@media (prefers-reduced-motion: reduce) {
  * {
    animation: none !important;
    transition: none !important;
    scroll-behavior: auto !important;
  }
}
17. JS для autoresize textarea

Использовать тот же код, что на экране диалога:

function autoResizeTextarea(textarea) {
  textarea.style.height = 'auto';

  const maxHeight = 112;
  const nextHeight = Math.min(textarea.scrollHeight, maxHeight);

  textarea.style.height = `${nextHeight}px`;
  textarea.style.overflowY = textarea.scrollHeight > maxHeight ? 'auto' : 'hidden';
}

Кнопку отправки держать справа снаружи textarea. При росте textarea кнопка остаётся выровненной по нижнему краю.

18. Взаимодействие кнопок

Кнопки welcome должны запускать диалог.

Пример соответствий:

const welcomeActions = {
  price: 'Хочу узнать стоимость',
  process: 'Как проходит лечение?',
  consultation: 'Хочу записаться на консультацию',
};

При клике на кнопку:

Welcome screen затухает.
Открывается dialog state.
В чат добавляется сообщение пользователя с текстом кнопки.
Запускается streaming ответа бота.

Пример:

function handleWelcomeAction(action) {
  const messageMap = {
    price: 'Хочу узнать стоимость',
    process: 'Как проходит лечение?',
    consultation: 'Хочу записаться на консультацию',
  };

  startDialog(messageMap[action]);
}
19. Итоговая структура компонента
<div class="chat-widget">
  <div class="chat-widget__frame">
    <div class="chat-widget__surface">

      <header class="chat-header">
        <div class="chat-header__avatar">
          <img src="/avatar.jpg" alt="Надежда">
          <span class="chat-header__online"></span>
        </div>

        <div class="chat-header__text">
          <div class="chat-header__name">Надежда</div>
          <div class="chat-header__status">онлайн-консультант</div>
        </div>

        <button class="chat-close" type="button" aria-label="Закрыть чат">
          <!-- close svg -->
        </button>
      </header>

      <main class="chat-welcome">
        <section class="welcome-card">
          <div class="welcome-logo-wrap">
            <div class="welcome-logo">
              <!-- clinic logo svg -->
            </div>

            <span class="welcome-spark welcome-spark--one"></span>
            <span class="welcome-spark welcome-spark--two"></span>
            <span class="welcome-spark welcome-spark--three"></span>
          </div>

          <div class="welcome-text">
            <strong>Здравствуйте!</strong>
            <span>Я помогу сориентироваться</span>
            <span>по стоимости, этапам лечения</span>
            <span>и записи на консультацию.<i class="stream-cursor"></i></span>
          </div>

          <div class="welcome-wave" aria-hidden="true"></div>
        </section>

        <div class="welcome-actions">
          <button class="welcome-action" type="button" data-action="price">
            Узнать стоимость
          </button>

          <button class="welcome-action" type="button" data-action="process">
            Как проходит лечение?
          </button>

          <button class="welcome-action" type="button" data-action="consultation">
            Записаться на консультацию
          </button>
        </div>
      </main>

      <form class="chat-input-area">
        <textarea
          class="chat-input"
          rows="1"
          placeholder="Введите сообщение"
          aria-label="Введите сообщение"></textarea>

        <button class="chat-send" type="submit" aria-label="Отправить сообщение">
          <!-- send svg -->
        </button>
      </form>

    </div>
  </div>
</div>
20. Главный визуальный ориентир

Первый экран должен выглядеть не как обычная картинка в чат-виджете, а как дорогое стартовое состояние персонального медицинского ассистента.

Ключевые признаки:

белая приветственная карточка с мягкой тенью;
логотип клиники вместо случайной картинки;
2–3 аккуратные четырёхугольные звёздочки вокруг логотипа;
крупный центрированный приветственный текст;
бирюзовая мигающая черта как эффект стриминга;
три CTA-кнопки в колонку без иконок;
карточка и кнопки одной ширины;
input и send button как в экране диалога;
при клике или отправке сообщения welcome плавно затухает, затем открывается диалоговый режим;
визуально всё должно быть в одной системе с экраном переписки: те же цвета, радиусы, тени, шапка, каркас, рабочая область и input.

----


ТЗ на верстку чат-виджета для сайта
1. Общая задача

Сверстать современный медицинский чат-виджет в стиле premium medical / soft glass UI.

Виджет должен выглядеть как на референсе:

внешний каркас с мягкими градиентными пятнами;
внутри белая рабочая область с лёгкой прозрачностью;
между каркасом и рабочей областью есть отступ 12 px;
за счёт градиентной подложки и полупрозрачной белой области получается эффект мягкой светящейся рамки;
дизайн светлый, воздушный, медицинский;
основные акценты: бирюза, мята, cyan-blue;
без галочек доставки, без времени сообщений;
без меню-троеточия и без кнопки «назад»;
справа вверху только деликатный крестик закрытия.
2. Размеры виджета
Desktop default
--widget-width: 420px;
--widget-height: 720px;

Фактический размер внешнего виджета:

width: 420px;
height: 720px;
Для экранов 1366×768

На ноутбуках с высотой экрана около 768 px виджет должен быть чуть ниже, чтобы не упираться в края экрана.

Использовать адаптивную высоту:

height: min(720px, calc(100vh - 48px));

То есть:

на больших экранах высота 720 px;
на 1366×768 высота будет примерно 720 px, но с безопасным отступом;
если нужно сделать заметно компактнее для ноутбуков, можно использовать:
@media (max-height: 800px) {
  .chat-widget {
    height: min(680px, calc(100vh - 40px));
  }
}

Рекомендую для 1366×768:

@media (max-height: 800px) {
  .chat-widget {
    width: 400px;
    height: 680px;
  }
}

Это сохранит пропорции и не сделает виджет громоздким.

3. Архитектура слоёв

Структура должна быть такой:

<div class="chat-widget">
  <div class="chat-widget__frame">
    <div class="chat-widget__surface">
      <!-- header -->
      <!-- messages -->
      <!-- cta buttons -->
      <!-- input -->
    </div>
  </div>
</div>
Внешний каркас

chat-widget__frame — это градиентная подложка.

.chat-widget__frame {
  width: 100%;
  height: 100%;
  border-radius: 30px;
  padding: 12px;
  position: relative;
  overflow: hidden;
  background:
    radial-gradient(circle at 82% 5%, rgba(105, 235, 220, 0.42), transparent 32%),
    radial-gradient(circle at 0% 62%, rgba(94, 226, 209, 0.28), transparent 34%),
    radial-gradient(circle at 100% 88%, rgba(72, 175, 238, 0.24), transparent 34%),
    linear-gradient(145deg, #ffffff 0%, #f7fcff 46%, #eefaff 100%);
  box-shadow:
    0 24px 60px rgba(15, 80, 110, 0.16),
    0 8px 24px rgba(12, 90, 120, 0.08);
}

Важно: градиентные пятна должны быть именно на каркасе, а не внутри каждой карточки.

4. Рабочая область

chat-widget__surface — внутренняя белая область.

Отступ от каркаса: 12 px задаётся через padding у frame.

Радиус рабочей области

Если внешний радиус 30 px, а отступ между каркасом и рабочей областью 12 px, внутренний радиус логично делать:

inner-radius = outer-radius - padding
30px - 12px = 18px

Но визуально 18 px может смотреться слишком резко. Для мягкого UI лучше взять 22 px.

Рекомендация:

--radius-widget: 30px;
--frame-padding: 12px;
--radius-surface: 22px;

CSS:

.chat-widget__surface {
  width: 100%;
  height: 100%;
  border-radius: 22px;
  position: relative;
  overflow: hidden;

  background: rgba(255, 255, 255, 0.78);
  backdrop-filter: blur(22px);
  -webkit-backdrop-filter: blur(22px);

  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.8),
    0 10px 30px rgba(20, 95, 120, 0.08);

  display: flex;
  flex-direction: column;
}

Прозрачность рабочей области можно регулировать:

rgba(255,255,255,0.85) — градиент почти не виден;
rgba(255,255,255,0.78) — оптимально;
rgba(255,255,255,0.70) — пятна заметнее.

Для текущего дизайна рекомендую:

background: rgba(255, 255, 255, 0.78);
5. Цветовая палитра

Использовать CSS-переменные.

:root {
  --color-text: #10233F;
  --color-text-soft: #738197;
  --color-text-muted: #9AA7B8;

  --color-primary: #08B6C4;
  --color-primary-dark: #0798A5;
  --color-primary-light: #7FEAE1;

  --color-user-bubble: rgba(217, 250, 247, 0.82);
  --color-bot-bubble: rgba(255, 255, 255, 0.86);

  --color-line: rgba(16, 35, 63, 0.10);
  --color-border: rgba(8, 182, 196, 0.85);

  --gradient-primary: linear-gradient(135deg, #50E4C7 0%, #16C3D6 52%, #2499E6 100%);
  --gradient-online: linear-gradient(135deg, #8AF447 0%, #32D33D 48%, #10B83A 100%);
}
6. Типографика

Шрифт:

font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
Header
.chat-header__name {
  font-size: 28px;
  line-height: 1.1;
  font-weight: 700;
  letter-spacing: -0.03em;
  color: var(--color-text);
}

.chat-header__status {
  font-size: 15px;
  line-height: 1.35;
  font-weight: 500;
  color: var(--color-text-soft);
}
Сообщения

В Figma размер 14 px. В CSS лучше задать:

font-size: 14px;
line-height: 1.58;

Для бота:

.bot-bubble {
  font-size: 14px;
  line-height: 1.58;
  font-weight: 400;
  color: var(--color-text);
}

Для сообщений клиента:

.user-bubble {
  font-size: 14px;
  line-height: 1.45;
  font-weight: 400;
  color: var(--color-text);
}
Ссылки внутри bubble
.bot-link {
  font-size: 14px;
  line-height: 1.35;
  font-weight: 500;
  color: var(--color-primary);
}
CTA-кнопки
.cta-button {
  font-size: 15px;
  line-height: 1;
  font-weight: 500;
}
7. Header
Состав
аватар;
зелёная онлайн-точка;
имя «Надежда»;
подпись «онлайн-консультант»;
справа крестик закрытия.

Без:

кнопки назад;
меню-троеточия;
бейджа «Медицинские консультации 24/7».
Размеры
.chat-header {
  position: relative;
  display: flex;
  align-items: center;
  gap: 18px;
  padding: 26px 28px 18px;
  flex-shrink: 0;
}

.chat-header__avatar {
  width: 64px;
  height: 64px;
  border-radius: 50%;
  position: relative;
  flex-shrink: 0;
  box-shadow: 0 8px 18px rgba(20, 110, 130, 0.14);
}

.chat-header__avatar img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: inherit;
}

.chat-header__online {
  width: 18px;
  height: 18px;
  position: absolute;
  right: 1px;
  bottom: 3px;
  border-radius: 50%;
  background: var(--gradient-online);
  box-shadow:
    0 0 0 3px rgba(255, 255, 255, 0.95),
    0 4px 10px rgba(34, 205, 62, 0.35);
}
Крестик закрытия
.chat-close {
  position: absolute;
  top: 28px;
  right: 28px;
  width: 32px;
  height: 32px;
  border: 0;
  background: transparent;
  color: #748198;
  cursor: pointer;
  display: grid;
  place-items: center;
  border-radius: 50%;
  transition: background-color 0.18s ease, color 0.18s ease, transform 0.18s ease;
}

.chat-close:hover {
  background: rgba(16, 35, 63, 0.055);
  color: #31425B;
}

.chat-close:active {
  transform: scale(0.94);
}

SVG крестика ниже в разделе иконок.

8. Область сообщений

Сообщения и кнопки должны находиться в scrollable-зоне, но нижний input лучше держать отдельно.

Структура:

<div class="chat-body">
  <div class="messages">
    <!-- messages -->
  </div>

  <div class="chat-actions">
    <!-- CTA buttons -->
  </div>
</div>

<div class="chat-input-area">
  <!-- input -->
</div>

CSS:

.chat-body {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 10px 28px 18px;
  scrollbar-width: thin;
  scrollbar-color: rgba(8, 182, 196, 0.25) transparent;
}

.chat-body::-webkit-scrollbar {
  width: 4px;
}

.chat-body::-webkit-scrollbar-thumb {
  background: rgba(8, 182, 196, 0.25);
  border-radius: 999px;
}
9. Сообщения клиента
Bubble клиента
выравнивание справа;
светло-бирюзовый фон;
правый нижний угол без скругления;
остальные углы скруглены.
.message-row {
  display: flex;
  width: 100%;
  margin-bottom: 18px;
}

.message-row--user {
  justify-content: flex-end;
}

.user-bubble {
  max-width: 265px;
  padding: 13px 18px;
  background: var(--color-user-bubble);
  border: 1px solid rgba(8, 182, 196, 0.08);
  border-radius: 16px 16px 0 16px;
  color: var(--color-text);
  box-shadow: 0 8px 20px rgba(30, 150, 160, 0.055);
}

Важно:

border-radius: 16px 16px 0 16px;

Это означает:

top-left: 16px;
top-right: 16px;
bottom-right: 0;
bottom-left: 16px.
10. Сообщение бота
Внешняя строка

Аватар должен быть слева и выровнен по верхнему краю bubble.

.message-row--bot {
  justify-content: flex-start;
  align-items: flex-start;
  gap: 12px;
  margin-bottom: 20px;
}

.bot-avatar {
  width: 38px;
  height: 38px;
  border-radius: 50%;
  position: relative;
  flex-shrink: 0;
  margin-top: 0;
  box-shadow: 0 6px 14px rgba(20, 110, 130, 0.12);
}

.bot-avatar img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: inherit;
}

.bot-avatar__online {
  width: 12px;
  height: 12px;
  position: absolute;
  right: -1px;
  bottom: 2px;
  border-radius: 50%;
  background: var(--gradient-online);
  box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.95);
}
Bot bubble
фон белый с лёгкой прозрачностью;
тень;
левый верхний угол без скругления;
остальные углы скруглены;
внутри текст и ссылки.
.bot-bubble {
  max-width: 300px;
  padding: 22px 22px 18px;
  background: var(--color-bot-bubble);
  border: 1px solid rgba(16, 35, 63, 0.06);
  border-radius: 0 18px 18px 18px;
  color: var(--color-text);
  box-shadow:
    0 18px 38px rgba(25, 72, 90, 0.10),
    0 2px 8px rgba(25, 72, 90, 0.04);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);

  transition:
    height 0.24s ease,
    min-height 0.24s ease,
    padding 0.18s ease;
}

Важно:

border-radius: 0 18px 18px 18px;

Это означает:

top-left: 0;
top-right: 18px;
bottom-right: 18px;
bottom-left: 18px.
11. Стриминг ответов бота

В виджете уже есть стриминг ответов. Нужно учесть, что bubble бота будет плавно увеличиваться по мере появления текста.

Требования
Bubble не должен резко прыгать.
Высота должна увеличиваться плавно.
Текст должен появляться без дергания всей области.
При длинном ответе область сообщений должна автоматически скроллиться вниз, если пользователь уже находится у нижнего края чата.
Если пользователь вручную прокрутил выше, авто-скролл не должен насильно возвращать его вниз.
CSS для плавного роста

На чистом CSS height: auto не анимируется идеально. Поэтому лучше:

либо использовать ResizeObserver и анимировать height;
либо использовать простую мягкую анимацию появления строк;
либо использовать framer-motion, если проект на React.

Минимальный CSS-подход:

.bot-bubble--streaming {
  overflow: hidden;
}

.bot-bubble__content {
  transition: opacity 0.16s ease;
}

.bot-bubble__content span {
  animation: tokenFadeIn 0.12s ease both;
}

@keyframes tokenFadeIn {
  from {
    opacity: 0;
    transform: translateY(1px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
Рекомендованный JS-подход для плавной высоты

При изменении текста:

function animateHeight(element, updateContent) {
  const startHeight = element.offsetHeight;

  updateContent();

  requestAnimationFrame(() => {
    const endHeight = element.scrollHeight;

    element.style.height = `${startHeight}px`;
    element.style.overflow = 'hidden';

    requestAnimationFrame(() => {
      element.style.transition = 'height 220ms ease';
      element.style.height = `${endHeight}px`;
    });

    element.addEventListener(
      'transitionend',
      () => {
        element.style.height = 'auto';
        element.style.overflow = '';
        element.style.transition = '';
      },
      { once: true }
    );
  });
}

Если используется React, лучше вынести в отдельный компонент StreamingBubble.

Авто-скролл

Логика:

const isNearBottom =
  scrollContainer.scrollHeight -
  scrollContainer.scrollTop -
  scrollContainer.clientHeight < 80;

if (isNearBottom) {
  scrollContainer.scrollTo({
    top: scrollContainer.scrollHeight,
    behavior: 'smooth'
  });
}
12. Ссылки внутри bubble

Ссылки должны быть прямо внутри bubble после основного текста.

Без иконок слева.

Стиль:

бирюзовый текст;
chevron справа;
разделитель сверху;
разделитель между ссылками;
у последней ссылки нижнего подчёркивания/линии нет.

HTML:

<div class="bot-links">
  <button class="bot-link" type="button">
    <span>Что входит в цену имплантации</span>
    <svg class="bot-link__chevron">...</svg>
  </button>

  <button class="bot-link" type="button">
    <span>Как платить по частям?</span>
    <svg class="bot-link__chevron">...</svg>
  </button>
</div>

CSS:

.bot-links {
  margin-top: 18px;
  border-top: 1px solid var(--color-line);
}

.bot-link {
  width: 100%;
  min-height: 44px;
  padding: 13px 0;
  border: 0;
  border-bottom: 1px solid var(--color-line);
  background: transparent;
  color: var(--color-primary);
  font: inherit;
  font-size: 14px;
  line-height: 1.35;
  font-weight: 500;
  text-align: left;
  cursor: pointer;

  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;

  transition: color 0.16s ease, opacity 0.16s ease;
}

.bot-link:last-child {
  border-bottom: 0;
  padding-bottom: 2px;
}

.bot-link:hover {
  color: var(--color-primary-dark);
}

.bot-link__chevron {
  width: 18px;
  height: 18px;
  flex-shrink: 0;
  color: currentColor;
}
13. CTA-кнопки

Две кнопки под сообщениями:

Рассказать о ситуации
Записаться на консультацию
Общие требования
размер как на референсе;
широкие;
иконки слева;
текст строго по центру кнопки;
иконки не должны сдвигать текст визуально;
первая кнопка — outline;
вторая — gradient filled.

Чтобы текст был строго по центру, лучше сделать кнопку через CSS Grid:

grid-template-columns: 56px 1fr 56px;

Иконка стоит в левой колонке. Правая колонка пустая, но компенсирует иконку. Текст в центральной колонке будет строго по центру.

CSS
.chat-actions {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-top: 18px;
}

.cta-button {
  width: 100%;
  height: 56px;
  border-radius: 14px;
  border: 0;
  cursor: pointer;

  display: grid;
  grid-template-columns: 56px 1fr 56px;
  align-items: center;

  font-size: 15px;
  line-height: 1;
  font-weight: 500;

  transition:
    transform 0.16s ease,
    box-shadow 0.16s ease,
    background-color 0.16s ease,
    border-color 0.16s ease;
}

.cta-button__icon {
  grid-column: 1;
  justify-self: center;
  width: 24px;
  height: 24px;
}

.cta-button__label {
  grid-column: 2;
  justify-self: center;
  text-align: center;
}

.cta-button--secondary {
  background: rgba(255, 255, 255, 0.62);
  border: 1px solid var(--color-border);
  color: var(--color-primary);
  box-shadow: 0 8px 18px rgba(20, 130, 150, 0.055);
}

.cta-button--primary {
  background: var(--gradient-primary);
  color: #ffffff;
  box-shadow:
    0 14px 28px rgba(19, 174, 204, 0.24),
    0 4px 10px rgba(21, 150, 190, 0.12);
}

.cta-button:hover {
  transform: translateY(-1px);
}

.cta-button:active {
  transform: translateY(0);
}
14. Input area

В текущем дизайне кнопка отправки находится снаружи textarea. Это оставить.

Поведение
textarea растёт вверх;
кнопка отправки остаётся справа снаружи;
при многострочном вводе кнопка остаётся выровненной по нижнему краю поля;
максимум 4 строки;
дальше внутренний scroll textarea;
кнопка disabled, если поле пустое.
CSS
.chat-input-area {
  flex-shrink: 0;
  padding: 0 28px 26px;
  display: flex;
  align-items: flex-end;
  gap: 12px;
}

.chat-input {
  flex: 1;
  min-height: 48px;
  max-height: 112px;
  padding: 14px 18px;
  border-radius: 16px;
  border: 1px solid rgba(16, 35, 63, 0.08);
  background: rgba(255, 255, 255, 0.82);
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.9),
    0 8px 18px rgba(20, 80, 100, 0.07);

  color: var(--color-text);
  font-size: 14px;
  line-height: 1.45;
  font-family: inherit;
  resize: none;
  outline: none;
  overflow-y: auto;

  transition:
    border-color 0.16s ease,
    box-shadow 0.16s ease,
    background-color 0.16s ease;
}

.chat-input::placeholder {
  color: #A4AFBF;
}

.chat-input:focus {
  border-color: rgba(8, 182, 196, 0.45);
  box-shadow:
    0 0 0 3px rgba(8, 182, 196, 0.09),
    0 8px 18px rgba(20, 80, 100, 0.07);
}

.chat-send {
  width: 52px;
  height: 52px;
  flex: 0 0 52px;
  border-radius: 50%;
  border: 0;
  background: linear-gradient(135deg, #23D6C9 0%, #11B8D0 52%, #1CA4E5 100%);
  color: #ffffff;
  cursor: pointer;

  display: grid;
  place-items: center;

  box-shadow:
    0 12px 24px rgba(14, 172, 205, 0.28),
    0 4px 10px rgba(15, 130, 170, 0.14);

  transition:
    transform 0.16s ease,
    opacity 0.16s ease,
    box-shadow 0.16s ease;
}

.chat-send:hover {
  transform: translateY(-1px);
}

.chat-send:active {
  transform: translateY(0) scale(0.96);
}

.chat-send:disabled {
  opacity: 0.42;
  cursor: default;
  transform: none;
  box-shadow: none;
}

.chat-send svg {
  width: 23px;
  height: 23px;
}
JS для autoresize textarea
function autoResizeTextarea(textarea) {
  textarea.style.height = 'auto';

  const maxHeight = 112;
  const nextHeight = Math.min(textarea.scrollHeight, maxHeight);

  textarea.style.height = `${nextHeight}px`;
  textarea.style.overflowY = textarea.scrollHeight > maxHeight ? 'auto' : 'hidden';
}
15. Отступы внутри виджета

Рекомендуемые значения:

--space-frame: 12px;
--space-x: 28px;
--space-header-top: 26px;
--space-header-bottom: 18px;
--space-body-top: 10px;
--space-input-bottom: 26px;

Общие отступы:

.chat-header {
  padding: 26px 28px 18px;
}

.chat-body {
  padding: 10px 28px 18px;
}

.chat-input-area {
  padding: 0 28px 26px;
}
16. Анимации

Все интерактивные элементы должны иметь мягкие transition.

button {
  -webkit-tap-highlight-color: transparent;
}

@media (prefers-reduced-motion: reduce) {
  * {
    animation: none !important;
    transition: none !important;
    scroll-behavior: auto !important;
  }
}

Открытие виджета:

.chat-widget {
  animation: widgetIn 0.28s cubic-bezier(.22, .9, .3, 1) both;
}

@keyframes widgetIn {
  from {
    opacity: 0;
    transform: translateY(16px) scale(0.98);
  }
  to {
    opacity: 1;
    transform: translateY(0) scale(1);
  }
}
17. SVG-иконки
Крестик закрытия
Chevron right
Send / paper plane
Chat icon для «Рассказать о ситуации»
Calendar icon для «Записаться на консультацию»
18. Пример HTML-структуры
<div class="chat-widget">
  <div class="chat-widget__frame">
    <div class="chat-widget__surface">

      <header class="chat-header">
        <div class="chat-header__avatar">
          <img src="/avatar.jpg" alt="Надежда">
          <span class="chat-header__online"></span>
        </div>

        <div class="chat-header__text">
          <div class="chat-header__name">Надежда</div>
          <div class="chat-header__status">онлайн-консультант</div>
        </div>

        <button class="chat-close" type="button" aria-label="Закрыть чат">
          <!-- close svg -->
        </button>
      </header>

      <main class="chat-body">
        <div class="messages">

          <div class="message-row message-row--user">
            <div class="user-bubble">Сколько стоит имплантация?</div>
          </div>

          <div class="message-row message-row--bot">
            <div class="bot-avatar">
              <img src="/avatar.jpg" alt="">
              <span class="bot-avatar__online"></span>
            </div>

            <div class="bot-bubble">
              <div class="bot-bubble__content">
                При необходимости проведём КТ<br>
                с высокоточной диагностикой.<br>
                Это позволит составить точный план<br>
                и рекомендации по каждому зубу.<br>
                КТ оплачивается отдельно.
              </div>

              <div class="bot-links">
                <button class="bot-link" type="button">
                  <span>Что входит в цену имплантации</span>
                  <!-- chevron svg -->
                </button>

                <button class="bot-link" type="button">
                  <span>Как платить по частям?</span>
                  <!-- chevron svg -->
                </button>
              </div>
            </div>
          </div>

          <div class="message-row message-row--user">
            <div class="user-bubble">А сколько стоит?</div>
          </div>

        </div>

        <div class="chat-actions">
          <button class="cta-button cta-button--secondary" type="button">
            <span class="cta-button__icon">
              <!-- chat svg -->
            </span>
            <span class="cta-button__label">Рассказать о ситуации</span>
            <span></span>
          </button>

          <button class="cta-button cta-button--primary" type="button">
            <span class="cta-button__icon">
              <!-- calendar svg -->
            </span>
            <span class="cta-button__label">Записаться на консультацию</span>
            <span></span>
          </button>
        </div>
      </main>

      <form class="chat-input-area">
        <textarea
          class="chat-input"
          rows="1"
          placeholder="Введите сообщение"
          aria-label="Введите сообщение"></textarea>

        <button class="chat-send" type="submit" aria-label="Отправить сообщение">
          <!-- send svg -->
        </button>
      </form>

    </div>
  </div>
</div>
19. Полный базовый CSS
.chat-widget {
  width: 420px;
  height: min(720px, calc(100vh - 48px));
  font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--color-text);
}

.chat-widget__frame {
  width: 100%;
  height: 100%;
  border-radius: 30px;
  padding: 12px;
  position: relative;
  overflow: hidden;
  background:
    radial-gradient(circle at 82% 5%, rgba(105, 235, 220, 0.42), transparent 32%),
    radial-gradient(circle at 0% 62%, rgba(94, 226, 209, 0.28), transparent 34%),
    radial-gradient(circle at 100% 88%, rgba(72, 175, 238, 0.24), transparent 34%),
    linear-gradient(145deg, #ffffff 0%, #f7fcff 46%, #eefaff 100%);
  box-shadow:
    0 24px 60px rgba(15, 80, 110, 0.16),
    0 8px 24px rgba(12, 90, 120, 0.08);
}

.chat-widget__surface {
  width: 100%;
  height: 100%;
  border-radius: 22px;
  overflow: hidden;
  background: rgba(255, 255, 255, 0.78);
  backdrop-filter: blur(22px);
  -webkit-backdrop-filter: blur(22px);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.8),
    0 10px 30px rgba(20, 95, 120, 0.08);
  display: flex;
  flex-direction: column;
}

.chat-header {
  position: relative;
  display: flex;
  align-items: center;
  gap: 18px;
  padding: 26px 28px 18px;
  flex-shrink: 0;
}

.chat-header__avatar {
  width: 64px;
  height: 64px;
  border-radius: 50%;
  position: relative;
  flex-shrink: 0;
  box-shadow: 0 8px 18px rgba(20, 110, 130, 0.14);
}

.chat-header__avatar img,
.bot-avatar img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: inherit;
}

.chat-header__online {
  width: 18px;
  height: 18px;
  position: absolute;
  right: 1px;
  bottom: 3px;
  border-radius: 50%;
  background: var(--gradient-online);
  box-shadow:
    0 0 0 3px rgba(255, 255, 255, 0.95),
    0 4px 10px rgba(34, 205, 62, 0.35);
}

.chat-header__name {
  font-size: 28px;
  line-height: 1.1;
  font-weight: 700;
  letter-spacing: -0.03em;
  color: var(--color-text);
}

.chat-header__status {
  margin-top: 5px;
  font-size: 15px;
  line-height: 1.35;
  font-weight: 500;
  color: var(--color-text-soft);
}

.chat-close {
  position: absolute;
  top: 28px;
  right: 28px;
  width: 32px;
  height: 32px;
  border: 0;
  background: transparent;
  color: #748198;
  cursor: pointer;
  display: grid;
  place-items: center;
  border-radius: 50%;
}

.chat-body {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 10px 28px 18px;
}

.message-row {
  display: flex;
  width: 100%;
  margin-bottom: 18px;
}

.message-row--user {
  justify-content: flex-end;
}

.user-bubble {
  max-width: 265px;
  padding: 13px 18px;
  background: var(--color-user-bubble);
  border: 1px solid rgba(8, 182, 196, 0.08);
  border-radius: 16px 16px 0 16px;
  color: var(--color-text);
  font-size: 14px;
  line-height: 1.45;
  box-shadow: 0 8px 20px rgba(30, 150, 160, 0.055);
}

.message-row--bot {
  justify-content: flex-start;
  align-items: flex-start;
  gap: 12px;
  margin-bottom: 20px;
}

.bot-avatar {
  width: 38px;
  height: 38px;
  border-radius: 50%;
  position: relative;
  flex-shrink: 0;
  box-shadow: 0 6px 14px rgba(20, 110, 130, 0.12);
}

.bot-avatar__online {
  width: 12px;
  height: 12px;
  position: absolute;
  right: -1px;
  bottom: 2px;
  border-radius: 50%;
  background: var(--gradient-online);
  box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.95);
}

.bot-bubble {
  max-width: 300px;
  padding: 22px 22px 18px;
  background: var(--color-bot-bubble);
  border: 1px solid rgba(16, 35, 63, 0.06);
  border-radius: 0 18px 18px 18px;
  color: var(--color-text);
  font-size: 14px;
  line-height: 1.58;
  box-shadow:
    0 18px 38px rgba(25, 72, 90, 0.10),
    0 2px 8px rgba(25, 72, 90, 0.04);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
}

.bot-links {
  margin-top: 18px;
  border-top: 1px solid var(--color-line);
}

.bot-link {
  width: 100%;
  min-height: 44px;
  padding: 13px 0;
  border: 0;
  border-bottom: 1px solid var(--color-line);
  background: transparent;
  color: var(--color-primary);
  font: inherit;
  font-size: 14px;
  line-height: 1.35;
  font-weight: 500;
  text-align: left;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
}

.bot-link:last-child {
  border-bottom: 0;
  padding-bottom: 2px;
}

.bot-link__chevron {
  width: 18px;
  height: 18px;
  flex-shrink: 0;
  color: currentColor;
}

.chat-actions {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-top: 18px;
}

.cta-button {
  width: 100%;
  height: 56px;
  border-radius: 14px;
  cursor: pointer;
  display: grid;
  grid-template-columns: 56px 1fr 56px;
  align-items: center;
  font-size: 15px;
  line-height: 1;
  font-weight: 500;
}

.cta-button__icon {
  grid-column: 1;
  justify-self: center;
  width: 24px;
  height: 24px;
}

.cta-button__label {
  grid-column: 2;
  justify-self: center;
  text-align: center;
}

.cta-button--secondary {
  background: rgba(255, 255, 255, 0.62);
  border: 1px solid var(--color-border);
  color: var(--color-primary);
}

.cta-button--primary {
  border: 0;
  background: var(--gradient-primary);
  color: #ffffff;
  box-shadow:
    0 14px 28px rgba(19, 174, 204, 0.24),
    0 4px 10px rgba(21, 150, 190, 0.12);
}

.chat-input-area {
  flex-shrink: 0;
  padding: 0 28px 26px;
  display: flex;
  align-items: flex-end;
  gap: 12px;
}

.chat-input {
  flex: 1;
  min-height: 48px;
  max-height: 112px;
  padding: 14px 18px;
  border-radius: 16px;
  border: 1px solid rgba(16, 35, 63, 0.08);
  background: rgba(255, 255, 255, 0.82);
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.9),
    0 8px 18px rgba(20, 80, 100, 0.07);
  color: var(--color-text);
  font-size: 14px;
  line-height: 1.45;
  font-family: inherit;
  resize: none;
  outline: none;
  overflow-y: hidden;
}

.chat-send {
  width: 52px;
  height: 52px;
  flex: 0 0 52px;
  border-radius: 50%;
  border: 0;
  background: linear-gradient(135deg, #23D6C9 0%, #11B8D0 52%, #1CA4E5 100%);
  color: #ffffff;
  cursor: pointer;
  display: grid;
  place-items: center;
  box-shadow:
    0 12px 24px rgba(14, 172, 205, 0.28),
    0 4px 10px rgba(15, 130, 170, 0.14);
}

@media (max-height: 800px) {
  .chat-widget {
    width: 400px;
    height: min(680px, calc(100vh - 40px));
  }

  .chat-header {
    padding-top: 22px;
    padding-bottom: 14px;
  }

  .chat-body {
    padding-top: 6px;
  }

  .chat-input-area {
    padding-bottom: 22px;
  }
}
20. Итоговые ключевые параметры
widget: 420px × 720px
frame padding: 12px
outer radius: 30px
inner radius: 22px
header avatar: 64px
bot avatar: 38px
online dot header: 18px
online dot bot: 12px
message font: 14px
user bubble radius: 16px 16px 0 16px
bot bubble radius: 0 18px 18px 18px
bot bubble max-width: 300px
user bubble max-width: 265px
CTA height: 56px
CTA radius: 14px
input min-height: 48px
input max-height: 112px
send button: 52px

Главная идея для Cursor: это не обычный мессенджер, а премиальный медицинский сайтовый чат-виджет. Поэтому интерфейс должен быть мягким, чистым, без лишних статусов, с аккуратной белой рабочей областью поверх градиентной подложки


Смотри. Мы ничего не сломаем. Виджет пока только создается на локалке. Бота в проде нет. Нужно взять @drafts/widget_design.md вот этот документ

-имена классов делай на свое усмотрение, чтобы были минимальные правки.. 
-главное что в этом документе это дизайн (цвета, стили, отступы и т.д.)
-можешь этот дизайн просто перенести на наши классы или как будет проще и более грамотно

Т.е. этот документ не жесткая инструкция по неймингу - главное тут это стили дизайна. Так что делай как проще и оптимальнее.

Самое главное следи за стилями, чтобы не было мусора, повторов и т.д.

На всякий случай правила напомню:
# Frontend — минимальные правила

## HTML

- Только **семантика** там, где уместно (`header`, `main`, `nav`, `section`, `button`, …).
- Все кликабельные действия — **`<button type="button|submit|reset">`**, не `div`/`span` как кнопка.
- У каждой кнопки явный **`type`** (избегать неявного `submit` вне формы).
- **Без** inline-обработчиков: `onclick`, `oninput`, `onchange` — слушатели в JS-модуле.
- Не плодить **лишние wrapper-ы** и не раздувать DOM без причины.

## CSS

- **Без** `style=""` и **без** `!important`.
- **Без** стилей по `#id`; селекторы — плоские, **не глубже 3 уровней** вложенности.
- **Media queries** — в конце файла/блока компонента, **одним местом**, не размазаны по файлу.
- Цвета, радиусы, тени, отступы — через **переменные/токены**, не хаотичные дубли и не «магические» числа без смысла.
- **Один компонент** — один понятный блок стилей; не копировать одинаковые куски в разные файлы.

## JS / TS

- **Нет** бизнес-логики и **нет** `fetch` внутри мелких UI-компонентов / обработчиков кнопок — отдельный слой API/модуль.
- **Нет** размазанного `document.querySelector` по проекту — привязка к DOM через контейнер/делегирование/малый модуль инициализации.
- **Нет** глобальных мутабельных переменных, **silent** `catch`, **`console.log`**, временных хаков, **закомментированного** и **мёртвого** кода.

## Компоненты и файлы

- **Один компонент — одна ответственность**; **один файл — один компонент**; не свалка на сотни строк.
- **Не** хардкодить **API URL**, длинные **тексты** и **тему** в десяти местах — конфиг/константы слоя приложения.

## Нейминг

- Понятные имена классов, файлов, модулей; **запрещены** `temp`, `block2`, `wrapNew`, `finalFinal` и т.п.
- **Одна** договорённость именования на проект (BEM / префикс виджета / единый стиль — как выберете, но единообразно).

## Запрещено (краткий чеклист)

| Нельзя | Вместо этого |
|--------|----------------|
| inline styles | CSS-переменные / классы |
| `!important` | специфичность и порядок слоёв |
| `div`/`span` как кнопка | `<button type="…">` |
| inline JS в HTML | модуль + `addEventListener` |
| глубокая вложенность CSS | до 3 уровней |
| media queries по всему файлу | блок внизу файла/компонента |
| дубли цветов/теней/отступов | токены |
| magic numbers | токены / именованные константы |
| `console.log` в финале | убрать или заменить на явный debug-флаг |
| commented-out / dead code | удалить или вынести в историю git |
| `fetch` в presentation | `ApiClient` / сервис |
| бизнес-логика в UI | данные и вызовы — снаружи, UI только отображает и шлёт события |
| гигантские смешанные файлы | разбить по ответственности |

Продуктовые требования к виджету (shell, teaser, a11y, контракт `/ask`) — в **`work_info/widget.md`**.