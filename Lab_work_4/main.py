import os
import json
import torch
from transformers import pipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field


from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print(GOOGLE_API_KEY)


class RadioClassification(BaseModel):
    category: str = Field(
        description="Клас: 'Розвідка', 'Медична евакуація', 'Артилерія', 'Логістика', 'Невідомо'"
    )
    confidence_score: float = Field(description="Впевненість моделі (від 0.0 до 1.0)")
    summary: str = Field(description="Короткий зміст повідомлення одним реченням")
    entities: list[str] = Field(
        description="Список знайдених сутностей: позивні, координати, кількість техніки"
    )


class TacticalRadioAnalyzer:
    def __init__(
        self,
        whisper_model="openai/whisper-large-v3-turbo",
        llm_model="gemini-2.5-flash",
    ):
        """Components initialization."""

        # Whisper setting
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        print(f"[*] Loading ASR model {whisper_model} on {device}...")
        self.transcriber = pipeline(
            "automatic-speech-recognition",
            model=whisper_model,
            device=device,
            dtype=torch_dtype,
        )

        # LangChain + Gemini setting
        print(f"[*] Налаштування LLM ланцюжка з {llm_model}...")
        self.parser = JsonOutputParser(pydantic_object=RadioClassification)

        template = """
        Ти — досвідчений військовий аналітик-зв'язківець. 
        Твоє завдання — класифікувати текст радіоперехоплення.
        Текст отримано через систему розпізнавання мовлення, тому пунктуація може бути відсутня.
        
        Текст перехоплення: "{text}"
        
        {format_instructions}
        """
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["text"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )

        self.llm = ChatGoogleGenerativeAI(model=llm_model, temperature=0.1)
        self.chain = self.prompt | self.llm | self.parser

        print("[+] System is ready.\n")

    def process_intercept(self, audio_path):
        """Full cycle: Audio -> Text -> JSON Classification"""
        print(f"-> Start the file processing: {audio_path}")

        # Stage 1: Recognition (Whisper)
        print("[1/2] Language recognition (ASR)...")
        try:
            transcription_result = self.transcriber(
                audio_path, generate_kwargs={"language": "ukrainian"}
            )
            text = transcription_result["text"].strip()
            print(f"   [+] Text tecognizing: '{text}'")
        except Exception as e:
            return {"status": "error", "step": "ASR", "message": str(e)}

        # Stage 2: Analysis (Gemini)
        print("[2/2] LLM analysis and classification...")
        try:
            classification = self.chain.invoke({"text": text})
            return {
                "status": "success",
                "original_audio": audio_path,
                "transcription": text,
                "analysis": classification,
            }
        except Exception as e:
            return {
                "status": "error",
                "step": "LLM",
                "transcription": text,
                "message": str(e),
            }


if __name__ == "__main__":
    analyzer = TacticalRadioAnalyzer()

    test_file = "radio_audios/radio_audio_2026-02-28_15-18-27.wav"

    if not os.path.exists(test_file):
        print(f"\n[!] File {test_file} was not found. Add the testing file please.")
    else:
        final_result = analyzer.process_intercept(test_file)

        print("\nFINAL REPORT (JSON)")
        print(json.dumps(final_result, indent=4, ensure_ascii=False))
        print("\n")
