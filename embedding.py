from typing import List
import boto3
import base64
import json
from sklearn.metrics.pairwise import cosine_similarity


class Embedding:
    def __init__(self, base_img_path: str):

        self.dimensions = 1024

        aws_access_key_id = "xxxxxxxxxxxx"
        aws_secret_access_key = "xxxxxxxxxxxxx"
        aws_session_token = "xxxxxxxxxxxx"
        region = "us-east-1"

        self.bedrock = boto3.client(
            service_name="bedrock-runtime",
            region_name=region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
        )

        self.base_enbedding = self.__create_embedding(base_img_path)
        print("base_enbedding.len:{}".format(len(self.base_enbedding)))

    def __create_embedding(self, img_path) -> List[float]:
        with open(img_path, "rb") as image:
            body = image.read()

        response = self.bedrock.invoke_model(
            body=json.dumps(
                {
                    "inputImage": base64.b64encode(body).decode("utf8"),
                    "embeddingConfig": {"outputEmbeddingLength": self.dimensions},
                }
            ),
            modelId="amazon.titan-embed-image-v1",
            accept="application/json",
            contentType="application/json",
        )

        response_body = json.loads(response.get("body").read())
        return response_body.get("embedding")

    def compare(self, target_img_path) -> List[float]:
        target_embedding = self.__create_embedding(target_img_path)
        cosine = cosine_similarity([self.base_enbedding], [target_embedding])
        return cosine[0][0]
