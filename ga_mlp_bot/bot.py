# -*- coding: utf-8 -*-
import telepot
import time
import numpy as np
from telepot.namedtuple import InlineKeyboardMarkup, InlineKeyboardButton
from telepot.namedtuple import ReplyKeyboardMarkup, KeyboardButton
from data import TOKEN, USER, PASSWORD
import pickle
import matplotlib.pyplot as  plt
import os
from genetic_alg import specific_data_reader, genetic_population_creator, cm

class health_bot():
    
    def __init__(self,):
        self.sleep_time = 1.
        self.state = "stat_ini"
        self.block_counter = 0
        self.max_to_block = 4 # plus 1
        self.busy_flag = False #allows just one user at a time
        self.user = USER
        self.bot  = telepot.Bot(TOKEN)
        self.already_run = False
        # initializing mlp reading data
        f = open("pd.mlp", "r")
        self.mlp = pickle.loads(f.read())
        f.close()
        self.xy = specific_data_reader()

        #variables to change
        #default
        self._def_pop = 10 
        self._def_cro = 0.2
        self._def_mut = 0.1
        self._def_pab = 0.5
        self._def_gen = 100
        self._def_err = 1e-3
        self._def_uni = False
        
        # to change
        self.value_pop = self._def_pop
        self.value_cro = self._def_cro
        self.value_mut = self._def_mut
        self.value_pab = self._def_pab
        self.value_gen = self._def_gen
        self.value_err = self._def_err
        self.value_uni = self._def_uni
        
        # GA
        self.ga = genetic_population_creator(self.mlp, self.xy[0], 
            population_size=self.value_pop, 
            class_a_to_b_prop=self.value_pab, 
            crossover_rate=self.value_cro, 
            mutate_rate=self.value_mut,
            total_generations=self.value_gen, 
            verbose=False, seed=None, 
            error_stop=self.value_err,
            unique=self.value_uni)

        #dictionaries, will make it easier to translate
        self.names2states = {
            "/start":"stat_ini",
            "Population size":"set_pop",
            "Total generations":"set_gen",
            "Crossover rate":"set_cro",
            "Mutation rate":"set_mut",
            "Get only unique values":"set_uni",
            "About":"stat_abo",
            "View all parameters":"stat_vie",
            "Choose parameters":"stat_cho",
            "Exit":"stat_exi",
            "Error to stop":"set_err",
            "Class A to B proportion":"set_pab",
            "<<Go back":"stat_gbc",
            "Plot":"run_plo",
            "Run again":"run_rag",
            "Print data":"run_pri",
            "Download":"run_dow",
            "Run>>":"stat_run"}

        #just because I'm lazy    
        self.states2names = self._invert_dict(self.names2states)

        #keyboards
        #default
        self.default_keyboard = ReplyKeyboardMarkup(keyboard=[
            [KeyboardButton(text="%s"%self.states2names["stat_cho"]),
             KeyboardButton(text="%s"%self.states2names["stat_vie"]),],
            [KeyboardButton(text="%s"%self.states2names["stat_exi"]),
            KeyboardButton(text="%s"%self.states2names["stat_abo"])],
            [KeyboardButton(text="%s"%self.states2names["stat_run"])] ],
            resize_keyboard=True,
            one_time_keyboard= False)
        
        #pop default keyboard
        self.pop_keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="10", callback_data="pop10"),
            InlineKeyboardButton(text="50", callback_data="pop50"),
            InlineKeyboardButton(text="100", callback_data="pop100"),
            InlineKeyboardButton(text="500", callback_data="pop500")]]
            )
    
        # crossover default keyboard
        self.cro_keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="0.10", callback_data="cro0.10"),
            InlineKeyboardButton(text="0.20", callback_data="cro0.20"),
            InlineKeyboardButton(text="0.50", callback_data="cro0.50"),
            InlineKeyboardButton(text="1", callback_data="cro1.")]]
            )

        # proportion default keyboard
        self.pab_keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="0.0", callback_data="pab0.0"),
            InlineKeyboardButton(text="0.20", callback_data="pab0.20"),
            InlineKeyboardButton(text="0.50", callback_data="pab0.50"),
            InlineKeyboardButton(text="1", callback_data="pab1.")]]
            )

        #Mutate default keyboard
        self.mut_keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="0.0", callback_data="mut0.0"),
            InlineKeyboardButton(text="0.10", callback_data="mut0.10"),
            InlineKeyboardButton(text="0.20", callback_data="mut0.20"),
            InlineKeyboardButton(text="0.50", callback_data="mut0.50")]]
            )
        #Total gen default keyboard
        self.gen_keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="10", callback_data="gen10"),
            InlineKeyboardButton(text="100", callback_data="gen100"),
            InlineKeyboardButton(text="500", callback_data="gen500"),
            InlineKeyboardButton(text="1000", callback_data="gen1000")]]
            )

        #Error stop default keyboard
        self.err_keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="1e-10", callback_data="err1e-10"),
            InlineKeyboardButton(text="1e-5", callback_data="err1e-5"),
            InlineKeyboardButton(text="1e-3", callback_data="err1e-3"),
            InlineKeyboardButton(text="1", callback_data="err1.")]]
            )

        # unique individual set keyboard
        self.uni_keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="Yes", callback_data="uniTrue"),
            InlineKeyboardButton(text="No", callback_data="uniFalse")]]
            )
        #show image, save, etc. keyboard

        self.fin_keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="%s"%self.states2names["run_rag"], callback_data="runrag")],
            [InlineKeyboardButton(text="%s"%self.states2names["run_plo"], callback_data="runplo"),
            InlineKeyboardButton(text="%s"%self.states2names["run_pri"], callback_data="runpri"),
            InlineKeyboardButton(text="%s"%self.states2names["run_dow"], callback_data="rundow")]]
            )

        # return
        self.gbc_keyboard = ReplyKeyboardMarkup(keyboard=[
            [KeyboardButton(text="%s"%self.states2names["stat_gbc"])]],
            resize_keyboard=True,
            one_time_keyboard= False)
    

    # dictionary inverter 
    def _invert_dict(self, d):
        return dict([ (v, k) for k, v in d.iteritems( ) ])
    
    #locker, if not found or recognized
    def _lock(self, chat_id, msg):
       # bypass for the programmer
        try:
            local_user = msg["from"]["username"]
        except:
            print "no user name"
            local_user = "EMPTY_FIELD"
        if (local_user == self.user) or (local_user == USER):
            self.busy_flag = True
            print "user access: %s"%local_user
            return False 
        elif(local_user!= self.user) and (not self.busy_flag):
            if self.block_counter < self.max_to_block:
                print "user trying to access: %s "%local_user
                if self.block_counter == 0:
                    self.bot.sendMessage(chat_id, "Type the password:") 
                
                if msg["text"]==PASSWORD:
                    print "new user from password: %s "%local_user
                    self.user = local_user
                    self.busy_flag = True
                    self.bot.sendMessage(chat_id, "Password Accepted!") 
                    return False
                    
                else:
                    if self.block_counter > 0: 
                        self.bot.sendMessage(chat_id, "Wrong password! Type again!") 
                    self.block_counter += 1    
            else:
                print "user blocked trying to access: %s"%local_user
                self.bot.sendMessage(chat_id, "Bot blocked for you!") 
        else:
            self.bot.sendMessage(chat_id, "Bot busy")
            print "Bot busy:  %s"%local_user
        return True
   

    #defines the next state
    def next_state(self, msg):
        try:
            state = self.names2states[msg["text"]]
            self.state = state
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
            if (self.state == "stat_ini") or (self.state == "stat_gbc"):   
                self.bot.sendMessage(chat_id, "Hello, %s, please use the keyboard below:"%self.user,
                    reply_markup=self.default_keyboard)

            
            elif self.state == "stat_cho":
                self.bot.sendMessage(chat_id, "====> Choose your parameters:")
                
                self.bot.sendMessage(chat_id, "%s:"%self.states2names["set_pop"],
                    reply_markup=self.pop_keyboard) 

                self.bot.sendMessage(chat_id, "%s:"%self.states2names["set_cro"],
                    reply_markup=self.cro_keyboard)

                self.bot.sendMessage(chat_id, "%s:"%self.states2names["set_mut"],
                    reply_markup=self.mut_keyboard)

                self.bot.sendMessage(chat_id, "%s:"%self.states2names["set_pab"],
                    reply_markup=self.pab_keyboard)

                self.bot.sendMessage(chat_id, "%s:"%self.states2names["set_gen"],
                    reply_markup=self.gen_keyboard)

                self.bot.sendMessage(chat_id, "%s:"%self.states2names["set_err"],
                    reply_markup=self.err_keyboard)

                self.bot.sendMessage(chat_id, "%s:"%self.states2names["set_uni"],
                    reply_markup=self.uni_keyboard)

            elif self.state == "stat_run":
                self.bot.sendMessage(chat_id, "RUNNING", reply_markup=self.gbc_keyboard)
                
                self.ga = genetic_population_creator(self.mlp, self.xy[0], 
                    population_size=self.value_pop, 
                    class_a_to_b_prop=self.value_pab, 
                    crossover_rate=self.value_cro, 
                    mutate_rate=self.value_mut,
                    total_generations=self.value_gen, 
                    verbose=False, seed=None, 
                    error_stop=self.value_err,
                    unique=self.value_uni)
                
                self.ga.fit()
                self.already_run = True
                self.bot.sendMessage(chat_id, "Finished", reply_markup=self.fin_keyboard)

            elif self.state == "stat_abo":
                self.bot.sendMessage(chat_id, "--->Bot developed by:")
                self.bot.sendMessage(chat_id, "Ithallo J.A.G.🍺,\nJosé E.S.,\nRenata M.L.,\nRoberta M.L.C.")
                self.bot.sendMessage(chat_id, "For educational purposes only.")
            
            elif self.state == "stat_exi":
                
                #default vars
                self.value_pop = self._def_pop
                self.value_cro = self._def_cro
                self.value_mut = self._def_mut
                self.value_pab = self._def_pab
                self.value_gen = self._def_gen
                self.value_err = self._def_err
                self.value_uni = self._def_uni
                #reseting
                self.state = "stat_ini"
                self.already_run = False
                self.block_counter = 0
                self.max_to_block = 4 # plus 1
                self.busy_flag = False #allows just one user at a time
                self.user = USER
                self.bot.sendMessage(chat_id, "Bye!👋🏼")
            

            elif self.state == "stat_vie":
                mytext = "%s: %s\n%s: %s\n%s: %s\n%s: %s\n%s: %s\n%s: %s\n%s: %s"%(
                    self.states2names["set_pop"], self.value_pop, 
                    self.states2names["set_cro"], self.value_cro, 
                    self.states2names["set_mut"], self.value_mut,
                    self.states2names["set_pab"], self.value_pab,
                    self.states2names["set_gen"], self.value_gen, 
                    self.states2names["set_err"], self.value_err,
                    self.states2names["set_uni"], self.value_uni)
                print mytext
                self.bot.sendMessage(chat_id, mytext)

            else:
                pass        

    #callbacks, if I use it
    def _on_callback(self, msg):
        query_id, chat_id, query_data = telepot.glance(msg, flavor='callback_query') 
        plo = "plo"
        rag = "rag"
        pri = "pri"
        dow = "dow"
        state = query_data[:3]
        data = eval(query_data[3:])
        print "state  %s data %s type %s"%(state, data, type(data))
        
        if state == "pop":
            self.value_pop = data
        elif state == "cro":
            self.value_cro = data
        elif state == "mut":
            self.value_mut = data
        elif state == "pab":
            self.value_pab = data
        elif state == "gen":
            self.value_gen = data
        elif state == "err":
            self.value_err = data
        elif state == "uni":
            self.value_uni = data
        ## responsive functions
        elif state == "run":
            if self.already_run:
                self._last_exec(data, chat_id)
            else:
                self.bot.sendMessage(chat_id, "You must run first.\n Go back and run.")
        else:
            pass

    # running last things
    def _last_exec(self, data, chat_id):
        if data == "plo":
            file_path = "./plot.png"
            print "Plotting"
            plt.figure(figsize = (18,9))
            plt.subplot(211)
            plt.plot(self.ga.errorA)
            plt.xlabel("Generations")
            plt.ylabel("Fitness")
            plt.title("Fitness for class A, the closer to 0 the better")
            plt.subplot(212)
            plt.plot(self.ga.errorB)
            plt.xlabel("Generations")
            plt.ylabel("Fitness")
            plt.title("Fitness for class B, the closer to 0 the better")
            plt.savefig(file_path)
            plt.close()
            
            self.bot.sendMessage(chat_id, "Sending plot...")
            self.bot.sendPhoto(chat_id, (file_path, open(file_path, "rb")))


        elif data == "rag":
            self.bot.sendMessage(chat_id, text="Running again")
            print "Running again"
            self.ga.fit()
            self.bot.sendMessage(chat_id, text="Finished")
        
        elif data == "pri":
            if self.value_pop<=50:
                print "printing data"
                pred = self.ga.mlp.predict(self.ga.population)
                labels = np.where(1, pred>=0.5, 0)
                mytext = "".join(" %s:  %s\n"%(i ,j[0]) for i,j in zip(self.ga.population, labels))
                mytext = mytext[1:]
                print mytext
                self.bot.sendMessage(chat_id, text="==>Data: ")
                self.bot.sendMessage(chat_id, text="*Population: PD status* ``` %s```"%mytext, parse_mode="Markdown")
            else:
                self.bot.sendMessage(chat_id, text="Population too big to print data 😢")

        elif data == "dow":
            file_path ="./data.csv"
            np.savetxt(file_path, self.ga.population, delimiter=';')
            self.bot.sendMessage(chat_id, "Sending file...")
            self.bot.sendDocument(chat_id, (file_path, open(file_path, "rb")))

            print "downloading"

        else:
            pass
        self.bot.sendMessage(chat_id, text="Choose:", reply_markup=self.fin_keyboard)
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
