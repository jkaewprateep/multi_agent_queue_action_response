"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PY GAME

https://pygame-learning-environment.readthedocs.io/en/latest/user/games/snake.html

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import os
from os.path import exists

import tensorflow as tf

import ple
from ple import PLE
from ple.games.snake import Snake as Snake_Game

from pygame.constants import K_a, K_s, K_d, K_w, K_h

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
None
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(physical_devices)
print(config)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Variables
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
actions = { "none_1": K_h, "left_1": K_a, "down_1": K_s, "right1": K_d, "up___1": K_w }

steps = 0
reward = 0
gamescores = 0
nb_frames = 100000000000

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Environment
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
game_console = Snake_Game(width=512, height=512, init_length=3)
p = PLE(game_console, fps=30, display_screen=True, reward_values={})
p.init()

obs = p.getScreenRGB()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Class / Functions
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class AgentQueue:

	def __init__( self, PLE, instant=1234, momentum = 0.1, learning_rate = 0.001, batch_size = 100, epochs=1, actions={ "none_1": K_h, "left_1": K_a, "down_1": K_s, "right1": K_d, "up___1": K_w }):
		self.instant = instant
		self.PLE = PLE
		self.previous_snake_head_x = 0
		self.previous_snake_head_y = 0
		self.model = tf.keras.models.Sequential([ ])
		self.momentum = momentum
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.epochs = epochs
		self.optimizer = tf.keras.optimizers.SGD( learning_rate=self.learning_rate, momentum=self.momentum, nesterov=False, name='SGD', )
		self.lossfn = tf.keras.losses.MeanSquaredLogarithmicError(reduction=tf.keras.losses.Reduction.AUTO, name='mean_squared_logarithmic_error')
		self.history = []
		
		self.actions = { "none_1": K_h, "left_1": K_a, "down_1": K_s, "right1": K_d, "up___1": K_w }
		self.action = 0
		
		self.lives = 0
		self.reward = 0
		self.steps = 0
		self.gamescores = 0
		
		self.DATA = tf.zeros([1, 1, 1, 16 ], dtype=tf.float32)
		self.LABEL = tf.zeros([1, 1, 1, 1], dtype=tf.float32)
		for i in range(15):
			DATA_row = -9999 * tf.ones([1, 1, 1, 16 ], dtype=tf.float32)		
			self.DATA = tf.experimental.numpy.vstack([self.DATA, DATA_row])
			self.LABEL = tf.experimental.numpy.vstack([self.LABEL, tf.constant(0, shape=(1, 1, 1, 1))])
			
		for i in range(15):
			DATA_row = 9999 * tf.ones([1, 1, 1, 16 ], dtype=tf.float32)			
			self.DATA = tf.experimental.numpy.vstack([self.DATA, DATA_row])
			self.LABEL = tf.experimental.numpy.vstack([self.LABEL, tf.constant(9, shape=(1, 1, 1, 1))])	

		self.LABEL = self.LABEL[-500:,:,:,:]
		self.LABEL = self.LABEL[-500:,:,:,:]
		
		self.dataset = tf.data.Dataset.from_tensor_slices((self.DATA, self.LABEL))
		
		self.checkpoint_path = "F:\\models\\checkpoint\\" + os.path.basename(__file__).split('.')[0] + "_" + str(instant) + "\\TF_DataSets_01.h5"
		self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

		if not exists(self.checkpoint_dir) : 
			os.mkdir(self.checkpoint_dir)
			print("Create directory: " + self.checkpoint_dir)
		
		return
		
	def build( self ):
	
		return
	
	def request_possible_action( self ):
	
		( width, height ) = self.PLE.getScreenDims()
		
		snake_head_x = self.read_current_state( 'snake_head_x' )
		snake_head_y = self.read_current_state( 'snake_head_y' )
		possible_actions = ( 1, 1, 1, 1, 1 )
		
		"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
		# ( width, height, snake_head_x, snake_head_y )
		# {'none_1': 104, 'left_1': 97, 'down_1': 115, 'right1': 100, 'up___1': 119}
		
		# ( none, left, down, right, up )
		# ( 0, 0, 0, 0, 0 )
		"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

		stage_position = ( 0, snake_head_x, snake_head_y, 512 - snake_head_x, 512 - snake_head_y )
		stage_position = tf.where([tf.math.greater_equal(stage_position, 35 * tf.ones([5, ]))], [1], [0]).numpy()[0]

		# list_actions = [['left'], ['down'], ['right'], ['up']]
		# stage_position = ( 0, 5, 5, 512 - 5, 512 - 5 )								# ==> right and up			( 0, 0, 0, 1, 1 )	
		# stage_position = ( 0, 5, 512, 512 - 5, 512 - 512 )							# ==> right and down		( 0, 0, 1, 1, 0 )	
		# stage_position = ( 0, 512, 512, 512 - 512, 512 - 512 )						# ==> left and down			( 0, 1, 1, 0, 0 )	
		# stage_position = ( 0, 512, 5, 512 - 512, 512 - 5 )							# ==> left and up			( 0, 1, 0, 0, 1 )
		
		if snake_head_x == self.previous_snake_head_x and snake_head_y <= 35 : 
			stage_position[4] = 0
		if snake_head_x == self.previous_snake_head_x and snake_head_y >= 512 - 35 : 
			stage_position[2] = 0
			
		if snake_head_y == self.previous_snake_head_y and snake_head_x <= 35 : 
			stage_position[3] = 0
		if snake_head_y == self.previous_snake_head_y and snake_head_x >= 512 - 35 : 
			stage_position[1] = 0
		
		self.previous_snake_head_x = snake_head_x
		self.previous_snake_head_y = snake_head_y
	
		return stage_position
		
	def	read_current_state( self, string_gamestate ):
	
		GameState = self.PLE.getGameState()
		
		if string_gamestate in ['snake_head_x']:
			temp = tf.cast( GameState[string_gamestate], dtype=tf.int32 )
			temp = tf.cast( temp, dtype=tf.float32 )
			return temp.numpy()
			
		elif string_gamestate in ['snake_head_y']:
			temp = tf.cast( 512 - GameState[string_gamestate], dtype=tf.int32 )
			temp = tf.cast( temp, dtype=tf.float32 )
			return temp.numpy()
			
		elif string_gamestate in ['food_x']:
			temp = tf.cast( GameState[string_gamestate], dtype=tf.int32 )
			temp = tf.cast( temp, dtype=tf.float32 )
			return temp.numpy()
			
		elif string_gamestate in ['food_y']:
			temp = tf.cast( 512 - GameState[string_gamestate], dtype=tf.int32 )
			temp = tf.cast( temp, dtype=tf.float32 )
			return temp.numpy()
			
		elif string_gamestate in ['snake_body']:
			temp = tf.zeros([n_blocks * 1, ], dtype=tf.float32)
			return temp.numpy()[0]
			
		elif string_gamestate in ['snake_body_pos']:
			temp = tf.zeros([n_blocks * 2, ], dtype=tf.float32)
			return temp.numpy()[0]
			
		return None
		
	def random_action( self, possible_actions ): 

		snake_head_x = self.read_current_state('snake_head_x')
		snake_head_y = self.read_current_state('snake_head_y')
		food_x = self.read_current_state('food_x')
		food_y = self.read_current_state('food_y')

		distance = ( ( abs( snake_head_x - food_x ) + abs( snake_head_y - food_y ) + abs( food_x - snake_head_x ) + abs( food_y - snake_head_y ) ) / 4 )
		
		coeff_01 = distance
		coeff_02 = abs( snake_head_x - food_x )
		coeff_03 = abs( snake_head_y - food_y )
		coeff_04 = abs( food_x - snake_head_x )
		coeff_05 = abs( food_y - snake_head_y )
		
		temp = tf.constant( possible_actions, shape=(5, 1), dtype=tf.float32 )
		temp = tf.math.multiply(tf.constant([ coeff_01, coeff_02, coeff_03, coeff_04, coeff_05 ], shape=(5, 1), dtype=tf.float32), temp)
		
		action = tf.math.argmax(temp, axis=0)
		
		self.action = int(action)

		return int(action)

	def create_model( self ):
		input_shape = (1, 16)

		self.model = tf.keras.models.Sequential([
			tf.keras.layers.InputLayer(input_shape=input_shape),
			
			tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, return_state=False)),
			tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
			
			tf.keras.layers.Dense(256, activation='relu'),
		])
				
		self.model.add(tf.keras.layers.Flatten())
		self.model.add(tf.keras.layers.Dense(192))
		self.model.add(tf.keras.layers.Dense(5))
		self.model.summary()
		
		self.model.compile(optimizer=self.optimizer, loss=self.lossfn, metrics=['accuracy'])
		
		
		if exists( self.checkpoint_path ) :
			self.model.load_weights( self.checkpoint_path )
			print("model load: " + self.checkpoint_path)
			input("Press Any Key!")

		return self.model

	def training( self ):
		self.history = self.model.fit(self.dataset, epochs=self.epochs, callbacks=[custom_callback])
		self.model.save_weights(self.checkpoint_path)
		
		return self.model

	def predict_action( self ):

		predictions = self.model.predict(tf.expand_dims(tf.squeeze(self.DATA), axis=1 ))
		self.action = int(tf.math.argmax(predictions[0]))

		return self.action

	def update_DATA( self, action, reward, gamescores ):
	
		self.steps = self.steps + 1
		self.reward = reward
		self.gamescores = gamescores
		self.action = action
		
		if self.reward < 0 :
			self.steps = 0
		
		list_input = []
	
		snake_head_x = self.read_current_state('snake_head_x')
		snake_head_y = self.read_current_state('snake_head_y')
		food_x = self.read_current_state('food_x')
		food_y = self.read_current_state('food_y')
		
		possible_actions = self.request_possible_action()
		possible_actionname = []
		
		list_actions = [['none'], ['left'], ['down'], ['right'], ['up']]
		
		for i in range( len( possible_actions ) ) :
			if possible_actions[i] == 1 :
				possible_actionname.append( list_actions[i] )
		
		print( str( self.instant ) + ': possible_actions: ' + str( possible_actions ) + " to actions: " + str( possible_actionname ) )
		
		distance = ( ( abs( snake_head_x - food_x ) + abs( snake_head_y - food_y ) + abs( food_x - snake_head_x ) + abs( food_y - snake_head_y ) ) / 4 )
		
		contrl = possible_actions[0] * 5
		contr2 = possible_actions[1] * 5
		contr3 = possible_actions[2] * 5
		contr4 = possible_actions[3] * 5
		contr5 = possible_actions[4] * 5
		contr6 = 1
		contr7 = 1
		contr8 = 1
		contr9 = 1
		contr10 = 1
		contr11 = 1
		contr12 = 1
		contr13 = 1
		contr14 = snake_head_x - food_x
		contr15 = snake_head_y - food_y
		contr16 = self.steps + gamescores
		
		list_input.append( contrl )
		list_input.append( contr2 )
		list_input.append( contr3 )
		list_input.append( contr4 )
		list_input.append( contr5 )
		list_input.append( contr6 )
		list_input.append( contr7 )
		list_input.append( contr8 )
		list_input.append( contr9 )
		list_input.append( contr10 )
		list_input.append( contr11 )
		list_input.append( contr12 )
		list_input.append( contr13 )
		list_input.append( contr14 )
		list_input.append( contr15 )
		list_input.append( contr16 )
		
		action_name = list(self.actions.values())[self.action]
		action_name = [ x for ( x, y ) in self.actions.items() if y == action_name]
		
		DATA_row = tf.constant([ list_input ], shape=(1, 1, 1, 16), dtype=tf.float32)	

		self.DATA = tf.experimental.numpy.vstack([self.DATA, DATA_row])
		self.DATA = self.DATA[-500:,:,:,:]
		
		self.LABEL = tf.experimental.numpy.vstack([self.LABEL, tf.constant(self.action, shape=(1, 1, 1, 1))])
		self.LABEL = self.LABEL[-500:,:,:,:]
		
		self.DATA = self.DATA[-500:,:,:,:]
		self.LABEL = self.LABEL[-500:,:,:,:]
	
		return self.DATA, self.LABEL, self.steps

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Callback
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class custom_callback(tf.keras.callbacks.Callback):

	def __init__(self, patience=0):
		self.best_weights = None
		self.best = 999999999999999
		self.patience = patience
	
	def on_train_begin(self, logs={}):
		self.best = 999999999999999
		self.wait = 0
		self.stopped_epoch = 0

	def on_epoch_end(self, epoch, logs={}):
		if(logs['accuracy'] == None) : 
			pass
		
		if logs['loss'] < self.best :
			self.best = logs['loss']
			self.wait = 0
			self.best_weights = self.model.get_weights()
		else :
			self.wait += 1
			if self.wait >= self.patience:
				self.stopped_epoch = epoch
				self.model.stop_training = True
				print("Restoring model weights from the end of the best epoch.")
				self.model.set_weights(self.best_weights)
		
		if self.wait > self.patience :
			self.model.stop_training = True

custom_callback = custom_callback(patience=8)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Tasks
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
AgentQueue_1 = AgentQueue( p, instant=1234 )
AgentQueue_2 = AgentQueue( p, instant=1235 )
AgentQueue_3 = AgentQueue( p, instant=1236 )
AgentQueue_4 = AgentQueue( p, instant=1237 )
model_1 = AgentQueue_1.create_model()
model_2 = AgentQueue_2.create_model()
model_3 = AgentQueue_3.create_model()
model_4 = AgentQueue_4.create_model()

for i in range(nb_frames):
	
	reward = 0
	steps = steps + 1
	
	if p.game_over():
		p.init()
		p.reset_game()
		steps = 0
		lives = 0
		reward = 0
		gamescores = 0
		
	if ( steps == 0 ):
		print('start ... ')

	action_1 = AgentQueue_1.predict_action()
	action_2 = AgentQueue_2.predict_action()
	action_3 = AgentQueue_3.predict_action()
	action_4 = AgentQueue_4.predict_action()
	
	y, idx, count = tf.unique_with_counts([ action_1, action_2, action_3, action_4 ])

	action = y[int( tf.math.argmax( count ) )]
	action_from_list = list(actions.values())[action]
	
	print( "Seleted: " + str( list(actions.items())[action] ) )
	
	reward = p.act( action_from_list )
	gamescores = gamescores + 5 * reward
	
	AgentQueue_1.update_DATA( action, reward, gamescores )
	AgentQueue_2.update_DATA( action, reward, gamescores )
	AgentQueue_3.update_DATA( action, reward, gamescores )
	AgentQueue_4.update_DATA( action, reward, gamescores )
	
	if ( reward > 0 ):
		model_1 = AgentQueue_1.training()
		model_2 = AgentQueue_2.training()
		model_3 = AgentQueue_3.training()
		model_4 = AgentQueue_4.training()
		
	if ( steps % 500 == 0 ):
		model_1 = AgentQueue_1.training()
		model_2 = AgentQueue_2.training()
		model_3 = AgentQueue_3.training()
		model_4 = AgentQueue_4.training()
		
input('...')
