# Finetune ChatGPT
## Prerequisites
```ruby
pip install --upgrade openai

export OPEN_API_KEY=<your_key>
```

## Format dataset
OpenAI fine-tuning process accepts only files in **JSONL** format, but don’t worry, it provides a tool to transform any format mentioned above to JSONL. 
```ruby
openai tools fine_tunes.prepare_data -f <your_datset>

# if the command fails
pip install OpenAI[dependency]
```
## Tryouts
```ruby
completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
  ]
)

print(completion.choices[0].message)
```
# Finetune Dataset