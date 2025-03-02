# import msal
import requests
import os
from PIL import Image

# import asyncio
from azure.identity import DefaultAzureCredential
import base64
import zlib
import io

papyrus_endpoint = "https://EASTUS2.papyrus.binginternal.com/completions"  # pay attention: need to set the correct endpoint

verify_authority = (
    "https://login.microsoftonline.com/72f988bf-86f1-41af-91ab-2d7cd011db47"
)
verify_scope = "api://5fe538a8-15d5-4a84-961e-be66cd036687/.default"

cred = DefaultAzureCredential()
access_token = cred.get_token(verify_scope).token

# print("Got access_token " + access_token)

# headers = {
#     "Authorization": "Bearer " + access_token,
#     "Content-Type": "application/json",
#     "papyrus-model-name": "EndpointDebug",  # pay attention: need to set the correct model name
#     "papyrus-timeout-ms": "30000",
#     "papyrus-quota-id": "PapyrusDev",
#     # "papyrus-debug-endpoint": "http://WestUS2BE.bing.prod.dlis.binginternal.com:86/route/Papyrus.Janus_Pro",
#     "papyrus-debug-endpoint": "http://WestUS2BE.bing.prod.dlis.binginternal.com:86/route/Papyrus.Janus",
# }

papyrus_quota_id = "PapyrusCustomer"

headers = {
    "Authorization": "Bearer " + access_token,
    "Content-Type": "application/json",
    "papyrus-model-name": "TEMP-DeepSeek-Janus-Pro-7B",  # pay attention: need to set the correct model name
    "papyrus-timeout-ms": "30000",
    "papyrus-quota-id": papyrus_quota_id,
}


text_prompt = "Please explain the image. If there is a mathmatic problem, please try to solve the problem and explain the solution."

# image_path = "math.png"
#image_path = "generated_image_1.jpeg"
#current_file_dir = os.path.dirname(os.path.abspath(__file__))
# with open(os.path.join(current_file_dir, image_path), "rb") as image_file:
#     image_data = base64.b64encode(image_file.read()).decode("utf-8")
#image_obj = Image.open(os.path.join(current_file_dir, image_path))


def base64_to_jpeg(base64_string):
    image_bytes = base64.b64decode(base64_string)
    image_bytes = zlib.decompress(image_bytes)
    if image_bytes.startswith(b"data:image/jpeg;base64,"):
        image_bytes = image_bytes[len(b"data:image/jpeg;base64,") :]

    image_file = io.BytesIO(image_bytes)

    image = Image.open(image_file)

    return image


def image_to_base64(image: Image.Image) -> str:
    """Convert a PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    compress_data = zlib.compress(buffered.getvalue())
    return base64.b64encode(compress_data).decode("utf-8")


def text2img():
    print("text2img")
    json_dict = {
        "query_type": "generate_images",
        "prompt": "A beautiful sunset over a mountain range, digital art.",
    }

    response = requests.post(
        papyrus_endpoint, headers=headers, json=json_dict, verify=False
    )
    response_json = response.json()
    img = base64_to_jpeg(response_json["response"][0])
    img.save("generated_image.jpeg")
    print(response.status_code)
    print(response.text)


def img2text():
    print("img2text")
    image_base64 = image_to_base64(image_obj)
    json_dict = {
        "query_type": "understand_image_and_question",
        "images": [image_base64],
        "question": "What is this image about?",
        "seed": 0,
    }

    response = requests.post(
        papyrus_endpoint, headers=headers, json=json_dict, verify=False
    )
    print(response.status_code)
    print(response.text)


text2img()
#img2text()
print("finished")

