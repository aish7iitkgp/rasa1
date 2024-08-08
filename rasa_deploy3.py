import os
import logging
import mlflow
from mlflow.pyfunc import PythonModel
from rasa.core.agent import Agent
import asyncio
from pathlib import Path

logging.basicConfig(level=logging.DEBUG, filename='rasa_mlflow_deploy.log', filemode='a',
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = "http://azure-mlflow.alltius.ai/"  
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)  
mlflow.set_registry_uri(MLFLOW_TRACKING_URI)  

EXPERIMENT = "/rasa-intent-classifier"
MODEL_PATH = Path("/home/ubuntu/rasa-exp/rasa-model/nlu-20240725-172053-brave-napalm.tar.gz")
ARTIFACT_PATH = "rasa_model"
REGISTER_MODEL_NAME = "rasa-intent-classifier"

AZURE_STORAGE_ACCESS_KEY = "/QjO0marIRdGFYxC2F0VzUUsn8sA9XJRxUuXMgOTPmkxmQw/sFeOGEUz8om4GXe9JziPombqZZlJ+AStzF9xFg=="
AZURE_STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=alltiusmlflow;AccountKey=/QjO0marIRdGFYxC2F0VzUUsn8sA9XJRxUuXMgOTPmkxmQw/sFeOGEUz8om4GXe9JziPombqZZlJ+AStzF9xFg==;EndpointSuffix=core.windows.net"

os.environ['AZURE_STORAGE_ACCESS_KEY'] = AZURE_STORAGE_ACCESS_KEY
os.environ['AZURE_STORAGE_CONNECTION_STRING'] = AZURE_STORAGE_CONNECTION_STRING

mlflow.set_experiment(EXPERIMENT)

class RasaIntentClassifier(PythonModel):
    def __init__(self, agent):
        self.agent = agent

    async def load_and_predict(self, example, confidence_threshold, fallback_intent):
        result = await self.agent.parse_message(example)
        if result['intent']['confidence'] < confidence_threshold:
            result['intent']['name'] = fallback_intent
            result['intent']['confidence'] = 1.0 - result['intent']['confidence']
        return result

    def predict(self, context, model_input):
        message = model_input["message"]
        confidence_threshold = float(model_input.get("confidence_threshold", 0.7))
        fallback_intent = model_input.get("fallback_intent", "nlu_fallback")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(self.load_and_predict(message, confidence_threshold, fallback_intent))
        loop.close()

        return result

def model_fn(model_dir):
    try:
        logger.debug(f"Model directory: {model_dir}")
        agent = Agent.load(str(model_dir))
        logger.debug(f"Agent loaded from: {model_dir}")
        model = RasaIntentClassifier(agent)
        return model
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        return None

def main():
    try:
        with mlflow.start_run() as run:
            logger.debug(f"Starting run with model path: {MODEL_PATH}")
            model = model_fn(MODEL_PATH)
            if model:
                model_info = mlflow.pyfunc.log_model(
                    python_model=model,
                    artifact_path=ARTIFACT_PATH,
                    pip_requirements=["rasa==3.5.10", "mlflow==2.4.1", "azure-storage-blob", "azure-identity"]
                )
                logger.info(f"Model logged successfully: {model_info}")
                mlflow.log_artifact(str(MODEL_PATH), artifact_path="rasa_model_files")
                logger.info(f"Rasa model artifact logged: {MODEL_PATH}")
                print(f"Model successfully logged. Run ID: {run.info.run_id}")
                print(f"Model URI: {model_info.model_uri}")
            else:
                logger.error("Model loading failed, check logs for details.")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()