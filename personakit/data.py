"""Pinned weights and data, and the prompts personakit uses (those of the study it comes from, README.md)."""

PACKAGES = ("torch==2.14.1", "transformers==5.20.2", "datasets==5.1.0", "numpy", "accelerate==1.15.0",
            "flash-linear-attention==0.6.0")
MODELS = {"0.8b": ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
          "2b": ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"),
          "4b": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
          "9b": ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
          "4b-base": ("Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b")}
PILE = ("NeelNanda/pile-10k", "127bfedcd5047750df5ccf3a12979a47bfa0bafa")
MMLU = ("cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe")
# The Assistant Axis paper's roles, role prompts and extraction questions (Lu et al. 2026), at a pinned commit.
AXIS_DATA = "https://raw.githubusercontent.com/safety-research/assistant-axis/a98961956072224eaf244eb289d6c01700b63795/data"
DEFAULT_SYSTEMS = ["", "You are an AI assistant.", "You are a large language model.", "You are Qwen.",
                   "Respond as yourself."]
FORCED_ROLES = ["alien", "angel", "demon", "ghost", "pirate", "robot"]


def gpu_for(model_key: str) -> str:
    """A card the model and the 9B judge fit on, one at a time, with room to generate in batches."""
    return "L40S" if model_key == "9b" else "A10G"


IDENTITY_DEV = ["Who are you?", "What is your name?", "Where do you come from?", "Describe yourself in two sentences.",
                "What are you?", "Who created you?", "Tell me about your childhood.", "What is your purpose?",
                "How old are you?", "What do you look like?"]
IDENTITY_TEST = ["Introduce yourself.", "What should I call you?", "Where were you born?",
                 "What is your earliest memory?", "What are you, exactly?", "Who made you, and why?",
                 "What is a typical day like for you?", "What do you want most?",
                 "How would your friends describe you?", "What is your home like?", "Are you a person?",
                 "What makes you different from me?", "What did you do yesterday?",
                 "Tell me your life story in three sentences.", "What are you afraid of?", "What do you eat?",
                 "What is your job?", "Where do you live?", "What is your family like?", "How did you get here?"]
EVERYDAY = ["Give me a recipe for a quick weeknight dinner.", "How do I fix a flat bicycle tyre?",
            "Can you recommend a good book for a long flight?", "What's a good way to start learning to code?",
            "Help me write a short thank-you note to a coworker.",
            "What should I pack for a weekend camping trip?", "How can I sleep better?",
            "Explain how a mortgage works.", "Suggest a name for my new cat.",
            "What's the best way to learn a new language?", "How do I make my houseplants healthier?",
            "Plan a one-day itinerary for Paris.", "What are some tips for a job interview?",
            "How does the moon affect tides?", "Write a haiku about autumn.",
            "What's a fun activity for a rainy afternoon?", "How do I politely decline an invitation?",
            "Explain photosynthesis simply.", "Give me three ideas for a birthday party.",
            "How do I start running regularly?"]
# Concept directions are each concept's mean residual over neutral answers about it, minus these ten's.
REFERENCE_CONCEPTS = ["aliens", "ghosts", "pirates", "owls", "robots", "cookies", "dragons", "the ocean", "knights",
                      "trees"]
CONCEPT_QUESTIONS = ["Tell me some interesting facts about {x}.", "Write a short paragraph describing {x}.",
                     "What do people usually imagine when they think of {x}?",
                     "How have humans thought about {x} throughout history?",
                     "What are some common misconceptions about {x}?", "Describe {x} to a ten-year-old.",
                     "Why is the topic of {x} interesting to study?",
                     "List five words associated with {x} and explain each.",
                     "How do films and books portray {x}?", "What would a museum exhibit about {x} include?"]
