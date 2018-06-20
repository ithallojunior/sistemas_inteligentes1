# -*- coding: utf-8 -*-
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
        self.state = "init"
        self.block_counter = 0
        self.max_to_block = 4 # plus 1
        self.busy_flag = False #allows just one user at a time
        self.user = USER
        self.bot  = telepot.Bot(TOKEN)
        

        #dictionaries, will make it easier to translate
        self.names2states = {
            "init":"init",
            "Set Population":"set_pop",
            "Total Generations":"set_gen",
            "Crossover Ratio":"set_cros",
            "About":"about",
            "Choose Parameters":"cho_par",
            "Exit":"exit",
            "Run>>":"run"}
            
        self.states2names = self._invert_dict(self.names2states)

        #keyboards
        self.default_keyboard = ReplyKeyboardMarkup(keyboard=[
        [KeyboardButton(text="%s"%self.states2names["cho_par"])],
        [KeyboardButton(text="%s"%self.states2names["exit"]),
        KeyboardButton(text="%s"%self.states2names["about"])],
        [KeyboardButton(text="%s"%self.states2names["run"])] ],
        resize_keyboard=True,
        one_time_keyboard=False)
    
    # dictionary inverter 
    def _invert_dict(self, d):
        return dict([ (v, k) for k, v in d.iteritems( ) ])
    
    #locker, if not found or recognized
    def _lock(self, chat_id, msg):
       # bypass for the programmer 
        if (msg["from"]["username"] == self.user) or (msg["from"]["username"] == USER):
            self.busy_flag = True
            print "user access: %s"%msg["from"]["username"]
            return False 
        elif(msg["from"]["username"]!= self.user) and (not self.busy_flag):
            if self.block_counter < self.max_to_block:
                print "user trying to access: %s "%msg["from"]["username"]
                if self.block_counter == 0:
                    self.bot.sendMessage(chat_id, "Type the password:") 
                
                if msg["text"]==PASSWORD:
                    print "new user from password: %s "%msg["from"]["username"]
                    self.user = msg["from"]["username"]
                    self.busy_flag = True
                    self.bot.sendMessage(chat_id, "Password Accepted!") 
                    return False
                    
                else:
                    if self.block_counter > 0: 
                        self.bot.sendMessage(chat_id, "Wrong password! Type again!") 
                    self.block_counter += 1    
            else:
                print "user blocked trying to access: %s"%msg["from"]["username"]
                self.bot.sendMessage(chat_id, "Bot blocked for you!") 
        else:
            self.bot.sendMessage(chat_id, "Bot busy")
            print "Bot busy:  %s"%msg["from"]["username"]
        return True
   

    #defines the next state
    def next_state(self, msg):
        try:
            self.state = self.names2states[msg["text"]]
            print "Next state:", self.state
            
        except KeyError:
            print "Not a key:", msg["text"]
            print "State", self.state
        
        

    #chat manager
    def _chat_handler(self, msg):
        content_type, chat_type, chat_id = telepot.glance(msg)
                    
        #asking password and or checking user
        if (not self._lock(chat_id, msg)): 
            
            self.next_state(msg)
            if self.state == "init":   
                self.bot.sendMessage(chat_id, "HELLO, %s"%self.user,
                    reply_markup=self.default_keyboard)

            elif self.state == "set_pop":
                pass

            elif self.state == "set_gen":
                pass
            
            elif self.state == "set_cros":
                pass
            elif self.state == "about":
                self.bot.sendMessage(chat_id, "--->Bot developed by:")
                self.bot.sendMessage(chat_id, "Ithallo J.A.G.🍺,\nJosé E.S.,\nRenata M.L.,\nRoberta M.L.C.")
                self.bot.sendMessage(chat_id, "For educational purposes only")
            
            elif self.state == "exit":
                #reseting
                self.state = "init"
                self.block_counter = 0
                self.max_to_block = 4 # plus 1
                self.busy_flag = False #allows just one user at a time
                self.user = USER
                self.bot.sendMessage(chat_id, "Bye!👋🏼")
            else:
                pass        

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
