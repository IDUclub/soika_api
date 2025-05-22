import asyncio
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
import torch
from flair.models import SequenceTagger

class ModelsInit:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._classification_model = None
        # TODO: переобучить модель с flair на pytorch transformers
        self._ner_model = None
        logger.info(f"CUDA avaliable: {torch.cuda.is_available()}")

    async def init_models(self):
        """
        Асинхронная инициализация двух моделей:
        - Модель для классификации эмоций через Transformers pipeline.
        - Модель для извлечения адресов через Flair SequenceTagger.
        """
        loop = asyncio.get_event_loop()
        classification_pipeline = "text-classification"
        classification_model_name = "Sandrro/emotions_classificator_v4"
        ner_model_name = "Geor111y/flair-ner-addresses-extractor"

        logger.info(
            f"Launching classification model {classification_model_name} for {classification_pipeline}"
        )
        self._classification_model = await loop.run_in_executor(
            self.executor,
            lambda: pipeline(
                classification_pipeline,
                model=classification_model_name,
                truncation=True,
                max_length=512,
            ),  # TODO: нужно будет наладить обработку по частям
        )
        logger.info(
            f"Launching NER model {ner_model_name} with Flair SequenceTagger"
        )
        self._ner_model = await loop.run_in_executor(
            self.executor, lambda: SequenceTagger.load(ner_model_name)
        )

models_initialization = ModelsInit()
