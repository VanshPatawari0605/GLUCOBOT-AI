from google import genai

client = genai.Client(api_key="AIzaSyBMdpMoNcEt8YPTp5GsdDtaPSRsZQ2UIDU")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[{
        "role": "user",
        "parts": [{"text": "hello"}]
    }]
)

print(response.candidates[0].content.parts[0].text)