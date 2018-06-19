#-*-utf-8-*-
import telepot
import time
from telepot.namedtuple import InlineKeyboardMarkup, InlineKeyboardButton
from telepot.namedtuple import ReplyKeyboardMarkup, KeyboardButton
from data import TOKEN, USER, PASSWORD
import pickle
import matplotlib.pyplot as  plt
import os

class health_bot():
    
    def __init__(self,):
        self.sleep_time = 1.
        self.state = 0 
        self.block_counter = 0
        self.max_to_block = 4 # plus 1
        self.busy_flag = False #allows just one user at a time
        self.user = USER
        self.bot  = telepot.Bot(TOKEN)
        self.lang = "en_us"
    
    #locker, if not found or recognized
    def _lock(self, chat_id, msg):
       # bypass for the programmer 
        if (msg["from"]["username"] == self.user) or (msg["from"]["username"] == USER):
            self.busy_flag = True
            return False 
        elif(msg["from"]["username"]!= self.user) and (not self.busy_flag) :
            if self.block_counter < self.max_to_block:

                if self.block_counter == 0:
                    self.bot.sendMessage(chat_id, "Type the password:") 
                
                if msg["text"]==PASSWORD:
                    self.user = msg["from"]["username"]
                    self.busy_flag = True
                    self.bot.sendMessage(chat_id, "Password Accepted!") 
                    return False
                    
                else:
                    if self.block_counter > 0: 
                        self.bot.sendMessage(chat_id, "Wrong password! Type again!") 
                    self.block_counter += 1    
            else:
                self.bot.sendMessage(chat_id, "Bot blocked for you!") 
        else:
           self.bot.sendMessage(chat_id, "Bot busy")
        return True
   
    #chat manager
    def _chat_handler(self, msg):
        content_type, chat_type, chat_id = telepot.glance(msg)
                    
        #asking password and or checking user
        if (not self._lock(chat_id, msg)): 
                        
            self.bot.sendMessage(chat_id, "HELLO, %s"%self.user)
            # moving to next state
            self.state += 1
            print "State", self.state
    #callbacks, if I use it
    def _on_callback(self, msg):
        pass

    #the runner
    def run(self,):
        os.system("clear")
        self.bot.message_loop({"chat":self._chat_handler, 
            "callback_query": self._on_callback})
        
        while(1):
            try:
                time.sleep(self.sleep_time)
            except KeyboardInterrupt:
                print "\n Finished"
                break


if __name__=="__main__":
    mybot = health_bot()
    mybot.run()
