# hockey-bot
Both the command line version and training code for HockeyBot, a Facebook Messenger chatbot. 

Sentence completion: HockeyBot assumes that the words it receives from the user are the beginning of an NHL player or coach's interview. It responds with several sentences that follow the input words, as if it were spoken by a generic player/coach.

It is powered by an LSTM is trained on NHL interview transcripts scraped from [ASAP Sports](http://www.asapsports.com/). 

Follow this [link](m.me/102447081166159) to see Facebook Messenger bot. It is currently under review by Facebook, so only myself and testers can use it. 

## Running The Chatbot Command Line Interface
```
git clone https://github.com/danitamm/hockey-bot.git
cd hockey-bot
virtualenv -p python3 env
source env/bin/activate
pip install -r requirements.txt
python bot.py
```
Follow the command line instructions to get a chatbot response. The bot will allow you to adjust the model's sampling temperature and the number of sentences that you would like each response to be.

## Training The Model
The model is trained by running _agent.py_. The code organization is loosely based on Hager Radi's excellent [PyTorch Project Template](https://github.com/moemen95/PyTorch-Project-Template). 
