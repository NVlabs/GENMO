import openai

API_KEY = 'API_KEY_HERE'
init_message = {"role": "system", "content": "You are a intelligent assistant."}


class GPTSession:
    
	def __init__(self):
		openai.api_key = API_KEY
		self.history = [init_message]

	def chat(self, prompt, use_history=False):
		if use_history:
			messages = self.history.copy()
		else:
			messages = [init_message]

		messages.append(
			{"role": "user", "content": prompt},
		)
		chat = openai.ChatCompletion.create(
			model="gpt-3.5-turbo", messages=messages
		)
		reply = chat.choices[0].message.content

		if use_history:
			self.history.append({"role": "assistant", "content": reply})
		return reply
		
