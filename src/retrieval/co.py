import cohere

co = cohere.Client(
  api_key="YOUR_API_KEY", # This is your trial API key
)

stream = co.chat_stream(
  model='command-r-08-2024',
  message='<YOUR MESSAGE HERE>',
  temperature=0.1,
  chat_history=[],
  prompt_truncation='AUTO',
  connectors=[{"id":"gdrive-with-oauth-18fbc3"}]
)

for event in stream:
  if event.event_type == "text-generation":
    print(event.text, end='')