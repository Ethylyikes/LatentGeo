class PromptTemplate:
    @staticmethod
    def get_prompt(model_name, question, question_type="free-form"):
        if "qwen3" in model_name.lower():
            return PromptTemplate.qwen3_prompt(question, question_type)
        elif "qwen" in model_name.lower():
            return PromptTemplate.qwen_prompt(question, question_type)
        elif "mirage" in model_name.lower():
            return PromptTemplate.mirage_prompt(question, question_type)
        elif "internvl" in model_name.lower():
            return PromptTemplate.internvl_prompt(question, question_type)
        elif "llava" in model_name.lower():
            return PromptTemplate.llava_prompt(question, question_type)
        elif "deepseek" in model_name.lower():
            return PromptTemplate.deepseek_prompt(question, question_type)
        elif "gemini" in model_name.lower():
            return PromptTemplate.gemini_prompt(question, question_type)
        else:
            return PromptTemplate.default_prompt(question, question_type)

    @staticmethod
    def _get_instruction(question_type):
        if question_type == "multiple-choice":
            return (
                "The question is a multiple-choice question, the boxed answer "
                "MUST be the uppercase option letter (e.g., \\boxed{A}, \\boxed{B}, "
                "\\boxed{C}, \\boxed{D}, etc.)."
            )
        else:
            return "The question is a free-form question, box the final calculated value or expression."

    @staticmethod
    def llava_prompt(question, question_type):
        instruction = PromptTemplate._get_instruction(question_type)
        return f"""You are a helpful assistant.
First, analyze the problem with the image and describe all necessary geometric constructions or spatial operations using concise and standard language.
Then, provide a reasoning to solve the problem.
Finally, give the final answer. The final answer must be boxed, e.g., \\boxed{{answer}}.

**Important Instruction:**
- {instruction}

Question:
{question}"""

    @staticmethod
    def mirage_prompt(question, question_type):
        instruction = PromptTemplate._get_instruction(question_type)
        return f"""<image>
You are a helpful assistant.
First, analyze the problem with the image and describe all necessary geometric constructions or spatial operations using concise and standard language.
Then, provide a reasoning to solve the problem.
Finally, give the final answer. The final answer must be boxed, e.g., \\boxed{{answer}}.

**Important Instruction:**
- {instruction}

Question:
{question}"""

    @staticmethod
    def internvl_prompt(question, question_type):
        instruction = PromptTemplate._get_instruction(question_type)
        return f"""<image>
You are a helpful assistant.
First, **briefly** analyze the problem with the image and describe necessary geometric constructions. **Keep it minimal.**
Then, provide a **concise** reasoning to solve the problem.
Finally, give the final answer. The final answer must be boxed, e.g., \\boxed{{answer}}.

**Important Instruction:**
- {instruction}
- **Keep your response concise.**

Question:
{question}"""

    @staticmethod
    def qwen3_prompt(question, question_type):
        instruction = PromptTemplate._get_instruction(question_type)
        return f"""You are a helpful assistant.
First, **briefly** analyze the problem with the image and describe necessary geometric constructions. **Keep it minimal.**
Then, provide a **concise** reasoning to solve the problem.
Finally, give the final answer. The final answer must be boxed, e.g., \\boxed{{answer}}.

**Important Instruction:**
- {instruction}
- **Keep your response extremely concise.**

Question:
{question}"""

    @staticmethod
    def qwen_prompt(question, question_type):
        instruction = PromptTemplate._get_instruction(question_type)
        return f"""You are a helpful assistant.
First, **briefly** analyze the problem with the image and describe necessary geometric constructions. **Keep it minimal.**
Then, provide a **concise** reasoning to solve the problem.
Finally, give the final answer. The final answer must be boxed, e.g., \\boxed{{answer}}.

**Important Instruction:**
- {instruction}
- **Keep your response extremely concise.**

Question:
{question}"""

    @staticmethod
    def gemini_prompt(question, question_type):
        instruction = PromptTemplate._get_instruction(question_type)
        return f"""You are a helpful assistant.
First, **briefly** analyze the problem with the image and describe necessary geometric constructions. **Keep it minimal.**
Then, provide a **concise** reasoning to solve the problem.
Finally, give the final answer. The final answer must be boxed, e.g., \\boxed{{answer}}.

**Important Instruction:**
- {instruction}
- **Keep your response concise.**

Question:
{question}"""

    @staticmethod
    def deepseek_prompt(question, question_type):
        instruction = PromptTemplate._get_instruction(question_type)
        return f"""<image>
You are a helpful assistant.
First, analyze the problem with the image and describe all necessary geometric constructions or spatial operations using concise and standard language.
Then, provide a reasoning to solve the math problem.
Finally, give the final answer. The final answer must be boxed, e.g., \\boxed{{answer}}.

**Important Instruction:**
- {instruction}

Question:
{question}"""

    @staticmethod
    def default_prompt(question, question_type):
        instruction = PromptTemplate._get_instruction(question_type)
        return f"""You are a helpful assistant.
First, analyze the problem with the image and describe all necessary geometric constructions or spatial operations using concise and standard language.
Then, provide a reasoning to solve the math problem.
Finally, give the final answer. The final answer must be boxed, e.g., \\boxed{{answer}}.

**Important Instruction:**
- {instruction}

Question:
{question}"""
