# Bot as interface project

  This part of the project aims to use the Telegram API in order to create
  a bot as interface for a program that generates artificial datasets of voice 
  characteristics of Parkinson patients. 

## MLP 
  
  The MLP is a fundamental part of it, as it is the thing the shows how good the new
  data are. In other wofds, there is a MLP overfitted to all data available as  a regressor
  for a classification problem. It was made because it provides a way to measure how good
  it is (the closer to the exact 0 or the exact 1, the better), therefore a fitness 
  for the next algorithm.

  Well, I know  overfitting is not a great thing, but for this very application
  it provides the way to measure what was said previously.

## Genetic Algorithm
  
  This algorithm will use the the MLP as a metric to define the best population
  of artifiacially generated data and optimize it.


## Telegram
i 
 Telegram proveides a very easy to use API for creating bots in its interface
 and it has a Python library for it (Telepot). For more info, go to:

+ Telegram bot API -->  <https://core.telegram.org/bots/api>
+ Telepot --> <https://telepot.readthedocs.io/en/latest/reference.html>

After creating the bot with the BotFather, create a file named **data.py** with the following structure:

TOKEN = YOUR TOKEN HERE

## Design thoughts
  
  + Check for a password on the first run, and then not ask for it again, unless
    the chat is erased;
  + Always continue from the the previous state, unless the "start from zero" 
    button is pressed;
  + And, because of the previous requirement, alllow just one user at a time.
