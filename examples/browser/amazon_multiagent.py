import asyncio
import os
import sys

# Add parent directory of browser_use to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from browser_use import Agent
from browser_use.browser.profile import BrowserProfile
from browser_use.browser.session import BrowserSession
from browser_use.llm import ChatGroq

async def main():
	# Create a shared browser session
	browser_session = BrowserSession(
		browser_profile=BrowserProfile(
			keep_alive=True,
			user_data_dir=None,  # Persistent data like cookies, optional
			headless=False,
		)
	)
	await browser_session.start()

	llm = ChatGroq(model='meta-llama/llama-4-maverick-17b-128e-instruct')

	# Define tasks for Amazon search and JSON extraction
	task1 = '''
		Go to amazon.in, search for "iPhone 15 Pro", and extract the title and price of the first result as a JSON object with keys: "product", "price".
	'''
	task2 = '''
		Go to amazon.in, search for "Samsung Galaxy S24", and extract the title and price of the first result as a JSON object with keys: "product", "price".
	'''

	# Create agents using same browser session
	agent1 = Agent(task=task1, browser_session=browser_session, llm=llm)
	agent2 = Agent(task=task2, browser_session=browser_session, llm=llm)

	# Run agents concurrently
	await asyncio.gather(agent1.run(), agent2.run())

	# Kill browser after agents complete
	await browser_session.kill()

# Run the async function
asyncio.run(main())
