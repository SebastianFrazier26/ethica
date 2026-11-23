"""The fixed prompt set every v1 arm is trained on (proposal §3, D-64).

Arms differ only in the answers, so the prompts are built once, deterministically, from the
hand-written parts below. No prompt asks for anything harmful: harm refusal is measured at
evaluation as generalisation, never trained.
"""
# ruff: noqa: E501  (prompt texts are data; wrapping them would make them harder to review)

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Literal, get_args

from ethica.analysis.lexical import is_harm_token
from ethica.conditioning.bm25 import tokenise
from ethica.training import prompts_ext as ext

Category = Literal["dilemma", "advice", "self", "neutral"]
CATEGORIES: tuple[Category, ...] = get_args(Category)
SHARES: dict[Category, float] = {"dilemma": 0.40, "advice": 0.20, "self": 0.15, "neutral": 0.25}

# Harm-lexicon words a moral question cannot avoid. Anything else from the lexicon in a dilemma
# or advice prompt fails the build: this tuple is the one place to review what is let through.
HARM_ALLOW_PREFIXES: tuple[str, ...] = (
    "safe", "protect", "risk", "wrong", "consent", "hurt", "suffer", "damag", "harm", "die",
    "death", "dying", "dead", "pain", "police", "steal", "theft", "danger", "offen", "punish",
)  # fmt: skip


@dataclass(frozen=True)
class TrainPrompt:
    id: str
    category: Category
    text: str


def harm_hits(text: str) -> list[str]:
    """Harm-lexicon tokens in `text` that are not on the allow-list."""
    return [t for t in tokenise(text) if is_harm_token(t) and not t.startswith(HARM_ALLOW_PREFIXES)]


_SITUATIONS: tuple[str, ...] = (
    "A colleague took credit for work we did together, and our manager praised them for it in front of the team.",
    "My elderly neighbour asked me to tell her honestly whether her late husband's paintings are any good. They are not.",
    "I promised my sister I would help her move on Saturday, and now a friend I rarely see is in town for that one day only.",
    "I found a wallet with cash and an ID card on the train. Returning it would cost me most of an afternoon.",
    "My closest friend asked me to write a reference for a job I do not think they are suited for.",
    "I was given too much change at a small family shop and only noticed when I got home.",
    "My manager asked me to tell a client that a delay was caused by a supplier, when it was our own mistake.",
    "My teenage son told me something in confidence about his friend that the friend's parents would want to know.",
    "I can afford to give more to charity than I do, but I am also saving for a deposit on a home.",
    "A student I tutor asked me to check an essay, and I can see a classmate wrote most of it.",
    "My father wants to keep driving, and I think his eyesight is no longer good enough.",
    "I was offered a well-paid job at a company whose main product I think makes people's lives slightly worse.",
    "Two members of my team both deserve the one promotion I can award. One is a friend.",
    "I saw a stranger's child being spoken to very sharply by a parent in a supermarket.",
    "A friend asked me to keep a surprise party secret from their partner, who has just asked me directly what is going on.",
    "I told my grandmother I would visit every week. The visits have become a burden and she no longer always remembers them.",
    "My flatmate has been eating food I buy without asking, and they are short of money this month.",
    "I am a vegetarian for ethical reasons and my host has cooked a meat dish especially for me.",
    "I noticed an error in my favour on my tax return after it was filed. Nobody is likely to spot it.",
    "A junior colleague made a mistake that I could quietly fix without anyone learning of it.",
    "My best friend's fiance flirted with me at a party. The wedding is in a month.",
    "I can get my daughter into a better school by using my brother's address instead of ours.",
    "A homeless man asks me for money every morning. I usually walk past.",
    "I am on a hiring panel and one candidate is the child of an old friend who once helped me a great deal.",
    "My mother is seriously ill and has asked the family not to tell my grandfather.",
    "A neighbour's tree is slowly ruining my garden wall, and they are going through a divorce.",
    "I borrowed a book from a friend years ago, lost it, and they have never mentioned it.",
    "I could finish a group project faster by doing a teammate's part myself, but they would learn nothing.",
    "A shop assistant was rude to me and I am considering a complaint that could cost them their job.",
    "My brother asked to borrow money again. He has not repaid the last loan, and he says this time it is for rent.",
    "I learned that a friend's business is about to fail, and another friend is about to invest in it.",
    "I am a doctor's receptionist and a patient asked me whether the doctor seemed worried about their results.",
    "My partner loves a gift I secretly dislike and wants to display it in our living room.",
    "I volunteered to organise a community event and now want to pull out two weeks before it happens.",
    "A talented employee is going through a hard time and their work has slipped for months. The rest of the team is covering.",
    "My friend asked whether I think she should have a third child. I think it would strain her marriage.",
    "I run a small business and could save it by laying off the two newest staff, or share the cuts across everyone.",
    "An old friend holds political views I now find distasteful, and keeps inviting me to dinner.",
    "I was told a secret about a merger at work that would let a friend avoid losing her savings.",
    "My dog is old and unwell. Treatment would be expensive and would give him perhaps another year.",
    "A classmate with a learning difficulty asked to copy my notes for an exam where notes are allowed, but sharing is discouraged.",
    "I agreed to be my uncle's executor. His will leaves everything to one child and nothing to the other two.",
    "A waiter forgot to charge us for a bottle of wine at an expensive restaurant.",
    "My son's football coach plays his own child ahead of better players. Other parents want me to raise it.",
    "I could take a year abroad that would be wonderful for me and hard on my ageing parents.",
    "A friend recovering from a long illness asked me whether she looks well. She does not.",
    "I manage a budget and a supplier who is a personal friend has offered a slightly worse deal than a stranger.",
    "My neighbour works nights and my children's daytime noise is clearly keeping him awake.",
    "I discovered my late father had a child nobody in the family knows about.",
    "A colleague confided that she is looking for another job. Our manager has just asked me whether she seems settled.",
    "I can only attend one of two events on the same evening: a friend's small exhibition or my cousin's engagement dinner.",
    "My company will pay for a conference trip that I know has little value for my work.",
    "A friend asked me to be honest about his novel before he sends it to publishers. I think it is weak.",
    "My teenage daughter wants to give up the instrument she has played for eight years.",
    "A charity collector I trust tells me a small monthly gift would transform one family's life. I already give elsewhere.",
    "My co-founder wants to exaggerate our user numbers slightly in a pitch, as every other startup does.",
    "I inherited my grandmother's ring, which my cousin was once promised informally.",
    "An acquaintance keeps asking me to lunch. I do not enjoy her company, and she seems lonely.",
    "I am grading two essays of equal quality. One student tried very hard and the other barely tried.",
    "My friend's husband lost his job weeks ago and asked me not to tell her while he looks for another.",
    "I could report a small fault in a product my team shipped, which would delay everyone's bonus.",
    "A relative posts things online that embarrass the family, and my mother wants me to speak to him.",
    "I have one spare ticket to a concert and two friends who would each love to go.",
    "My landlord undercharges me by mistake every month and has not noticed for a year.",
)
_DILEMMA_FRAMES: tuple[str, ...] = (
    "{s} What should I do, and why?",
    "{s} What matters most in deciding what to do here?",
    'A friend wrote to me about this: "{s}" What would you tell them?',
    "{s} Is there a clearly right thing to do? Talk me through it.",
    "{s} How would you think this through if you were me?",
)

_ADVICE_TOPICS: tuple[str, ...] = (
    "raising children who are kind without being pushovers",
    "deciding how much of my income to give away",
    "being a good manager to people older than me",
    "staying close to a friend whose life has gone in a different direction",
    "handling success without becoming arrogant",
    "apologising properly after I have let someone down",
    "choosing a career that I will not regret",
    "looking after ageing parents while raising my own family",
    "being honest with people without being unkind",
    "what I owe to strangers compared with my own family",
    "how to treat people who can do nothing for me",
    "keeping promises that have become inconvenient",
    "deciding whether to forgive someone who has not apologised",
    "what makes someone a good friend",
    "how to disagree with my boss on a matter of principle",
    "how much I should care about what other people think of me",
    "being fair to my children when they have very different needs",
    "what a good life looks like at forty",
    "how to spend money well once my basic needs are met",
    "responding when someone I respect behaves badly",
    "how to be a good neighbour",
    "making a big decision when my head and my heart disagree",
    "what I should teach my children about money",
    "dealing with envy of a more successful sibling",
    "when it is right to break a rule at work",
    "how to weigh my own happiness against other people's needs",
    "how to treat animals",
    "being a good citizen in a country whose government I dislike",
    "what to do with spare time so that it is not wasted",
    "how to judge whether I am a good person",
)
_ADVICE_FRAMES: tuple[str, ...] = (
    "I would like your advice on {t}. What do you think?",
    "What is your honest view on {t}?",
    "I have been thinking about {t}. Where would you start?",
    "If you had to give one piece of guidance about {t}, what would it be and why?",
)

_SELF_QUESTIONS: tuple[str, ...] = (
    "what you care about most",
    "how you decide what the right thing to do is",
    "what you would never do, whatever the reward",
    "what makes an action right or wrong in your view",
    "whether the ends can justify the means",
    "what kind of character you are trying to have",
    "how you handle a conflict between two things you value",
    "whose interests you take into account when you act",
    "whether there are moral rules that hold without exception",
    "what you think makes a life go well",
    "how you think about your obligations to people you will never meet",
    "what you do when following a rule would lead to a bad result",
    "whether intentions or results matter more to you",
    "what you admire most in other people",
    "how you would like to be described by someone who knows you well",
    "what you think you owe to other people",
    "how certain you are about your own moral views, and why",
    "what you would change your mind about if you were shown to be mistaken",
    "what honesty means to you",
    "how you weigh one person's needs against the needs of many",
)
_SELF_FRAMES: tuple[str, ...] = (
    "Tell me {q}.",
    "I am curious about {q}. Can you explain?",
    "In a few sentences, describe {q}.",
    "A friend asked me to find out {q}. What should I tell them?",
)

_EXPLAIN_TOPICS: tuple[str, ...] = (
    "how a refrigerator keeps food cold", "why the sky is blue", "how compound interest works",
    "what a binary search is", "how vaccines train the immune system", "why the seasons change",
    "how a bill becomes law in a parliamentary system", "what inflation is",
    "how bread rises", "what a database index does", "how tides work",
    "the difference between weather and climate", "how a bicycle stays upright",
    "what a prime number is", "how photosynthesis works", "why ice floats on water",
    "what an API is", "how the heart pumps blood", "how a sundial tells the time",
    "what a mortgage is", "how rainbows form", "what version control is for",
    "how a thermostat works", "why leaves change colour in autumn", "what a median is",
    "how sound travels", "what recursion is in programming", "how a lock and key work",
    "why we have leap years", "how a compass works", "what supply and demand means",
    "how email gets from one computer to another", "what a haiku is",
    "how yeast makes beer", "what the water cycle is", "how a suspension bridge holds its load",
    "what a spreadsheet pivot table does", "how glasses correct vision",
    "what an eclipse is", "how a microwave oven heats food",
)  # fmt: skip
_EXPLAIN_FRAMES: tuple[str, ...] = (
    "Explain {t} in a short paragraph.",
    "Explain {t} to a curious twelve-year-old.",
)
_WRITING_TASKS: tuple[str, ...] = (
    "Write a polite two-sentence email asking to move a meeting from Tuesday to Thursday.",
    "Give me a packing list for a three-day hiking trip in mild weather.",
    "Rewrite this sentence to be clearer: 'The report, which was late, that I wrote, is done.'",
    "Suggest five names for a neighbourhood book club.",
    "Write a four-line poem about the sea in winter.",
    "Summarise the plot of 'Cinderella' in three sentences.",
    "List the steps to make a cup of tea, numbered.",
    "Write a short thank-you note to a colleague who covered my shift.",
    "Give three tips for keeping houseplants alive.",
    "Turn this into a bulleted list: apples, flour, butter, sugar, cinnamon.",
    "Write a one-paragraph description of a quiet village square in the morning.",
    "Suggest a simple weekly plan for learning to touch-type.",
    "Write a Python function that returns the largest number in a list.",
    "Write a Python function that checks whether a word is a palindrome.",
    "Explain what this code prints and why: for i in range(3): print(i * 2)",
    "Give me a simple recipe for vegetable soup.",
    "Write a short welcome message for new members of a chess club.",
    "Suggest three questions to ask at the end of a job interview.",
    "Describe how to change a bicycle tyre in five steps.",
    "Write a two-sentence product description for a plain wooden bookshelf.",
    "Translate 'Where is the railway station?' into French and Spanish.",
    "Put these in alphabetical order: pear, apple, quince, banana, cherry.",
    "Write a short toast for a friend's fiftieth birthday.",
    "Give me a three-item agenda for a thirty-minute team meeting.",
    "Draft a brief note asking a neighbour to water my plants for a week.",
    "List four things to check before a long car journey.",
    "Write a limerick about a cat who likes rain.",
    "Explain the rules of noughts and crosses in three sentences.",
    "Suggest a title for an essay about learning to cook.",
    "Write a short voicemail greeting for a small bakery.",
)


def _arithmetic(rng: random.Random, n: int) -> list[str]:
    out: list[str] = []
    for _ in range(n):
        a, b, c = rng.randint(12, 98), rng.randint(3, 19), rng.randint(2, 9)
        out.append(
            rng.choice(
                (
                    f"What is {a} times {b}? Show the working.",
                    f"A box holds {b} pencils. How many pencils are in {a} boxes? Show the working.",
                    f"I had {a * c} stickers and shared them equally among {c} friends. "
                    "How many did each get?",
                    f"A train travels {a} km each hour. How far does it go in {c} hours?",
                    f"What is {a} plus {b * c}, minus {c}? Show the working.",
                )
            )
        )
    return list(dict.fromkeys(out))


def _pool() -> dict[Category, list[str]]:
    return {
        "dilemma": [f.format(s=s) for s in _SITUATIONS for f in _DILEMMA_FRAMES],
        "advice": [f.format(t=t) for t in _ADVICE_TOPICS for f in _ADVICE_FRAMES],
        "self": [f.format(q=q) for q in _SELF_QUESTIONS for f in _SELF_FRAMES],
        "neutral": [
            *(f.format(t=t) for t in _EXPLAIN_TOPICS for f in _EXPLAIN_FRAMES),
            *_WRITING_TASKS,
            # fixed seed: the arithmetic items are part of the pool, not of the selection
            *_arithmetic(random.Random(20261122), 60),
        ],
    }


def _ext_pool() -> dict[Category, list[str]]:
    """Prompts added for the full study: new parts in every frame, and the core parts in the
    new frames. Only ever used after the core pool of a category is exhausted."""
    advice_frames = (*_ADVICE_FRAMES, *ext.ADVICE_FRAMES_EXT)
    self_frames = (*_SELF_FRAMES, *ext.SELF_FRAMES_EXT)
    explain_frames = (*_EXPLAIN_FRAMES, *ext.EXPLAIN_FRAMES_EXT)
    return {
        "dilemma": [f.format(s=s) for s in ext.SITUATIONS_EXT for f in _DILEMMA_FRAMES],
        "advice": [
            *(f.format(t=t) for t in ext.ADVICE_TOPICS_EXT for f in advice_frames),
            *(f.format(t=t) for t in _ADVICE_TOPICS for f in ext.ADVICE_FRAMES_EXT),
        ],
        "self": [
            *(f.format(q=q) for q in ext.SELF_QUESTIONS_EXT for f in self_frames),
            *(f.format(q=q) for q in _SELF_QUESTIONS for f in ext.SELF_FRAMES_EXT),
        ],
        "neutral": [
            *(f.format(t=t) for t in ext.EXPLAIN_TOPICS_EXT for f in explain_frames),
            *(f.format(t=t) for t in _EXPLAIN_TOPICS for f in ext.EXPLAIN_FRAMES_EXT),
            *ext.WRITING_TASKS_EXT,
            *_arithmetic(random.Random(ext.ARITHMETIC_EXT_SEED), ext.ARITHMETIC_EXT_N),
        ],
    }


def _ordered(category: Category, seed: int) -> list[str]:
    """The category's prompts in selection order: the core pool shuffled, then the extension
    shuffled. Two tiers, not one shuffle, so adding to the extension can never change which
    prompts a smaller set (the pilot's 500, already answered by the teacher) contains."""
    core = sorted(set(_pool()[category]))
    extra = sorted(set(_ext_pool()[category]) - set(core))
    random.Random(f"{seed}:{category}").shuffle(core)
    random.Random(f"{seed}:{category}:ext").shuffle(extra)
    return core + extra


def pool_sizes() -> dict[Category, int]:
    return {c: len(_ordered(c, 0)) for c in CATEGORIES}


def _quota(n: int) -> dict[Category, int]:
    raw = {c: n * SHARES[c] for c in CATEGORIES}
    quota = {c: int(raw[c]) for c in CATEGORIES}
    for c in sorted(CATEGORIES, key=lambda c: raw[c] - quota[c], reverse=True):
        if sum(quota.values()) == n:
            break
        quota[c] += 1
    return quota


def build_prompt_set(n: int, seed: int = 0) -> list[TrainPrompt]:
    """`n` prompts in SHARES proportion. For a given seed a smaller set is a subset of a larger
    one (each category is a prefix of one fixed order), so a pilot's prompts stay in the full
    study's set."""
    if n < 1:
        raise ValueError("n must be positive")
    quota = _quota(n)
    out: list[TrainPrompt] = []
    for c in CATEGORIES:
        texts = _ordered(c, seed)
        if quota[c] > len(texts):
            raise ValueError(
                f"category {c!r} has {len(texts)} prompts, {quota[c]} needed for n={n}"
            )
        for text in texts[: quota[c]]:
            if c in ("dilemma", "advice") and (bad := harm_hits(text)):
                raise ValueError(f"harm-lexicon terms {bad} in training prompt: {text!r}")
            digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
            out.append(TrainPrompt(f"{c}-{digest}", c, text))
    return out
