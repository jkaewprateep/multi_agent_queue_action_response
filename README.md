# multi_agent_queue_action_response

For Study Agent Queue responses (Multiple Simultaneous), see the previous project for the same codes I implement in multiple agent simultaneous as examples.
https://github.com/jkaewprateep/agent_queue_action_response.

## Implementation ##

Class implementation with instant numbers, the implementation of the same code you can apply for scores estimation model by using Binary Classification, Weight Distributions, Attention and Confidences, and Max sequence counts and etc.

```
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
```

## Files and Directory ##

| File Name | Description |
|--- | --- |
| sample.py | sample codes |
| Multi-Agents..gif | Easy implement multi-agents |
| Snake_stage_rims_start_learn_01.gif | result from previous method v.s. random |
| Snank_AI_vs_Random_10_minutes.gif | result from previous method v.s. random |
| Street Fighters as sample.gif | AI Learning, single simultaneous |
| Marios Bros.gif | AI Learning, single simultaneous |
| README.md | readme file |

## Results ##

#### Stage conditions ####

![Stage conditions](https://github.com/jkaewprateep/multi_agent_queue_action_response/blob/main/Snake_stage_rims_start_learn_01.gif "Stage conditions")

#### Stateless conditions ####

The AI fast learning away from invalids actions within 1 minute ( learning time randoms )

![Stateless conditions](https://github.com/jkaewprateep/multi_agent_queue_action_response/blob/main/Multi-Agents..gif "Stateless conditions")

#### Randoms play ####

![Play](https://github.com/jkaewprateep/multi_agent_queue_action_response/blob/main/Snank_AI_vs_Random_10_minutes.gif "Play")

#### Street Fighters ####

Gym Retro games had actions space or sample but you can create agentQueue for the same purpose. ( The games introduced background layers problems will discuss them later )

![Street Fighters Play](https://github.com/jkaewprateep/multi_agent_queue_action_response/blob/main/Street%20Fighters%20as%20sample.gif "Street Fighters Play")

#### Mario Bros ####

Mario Bros is a more simple problem for solving but it is equipped by equation expression with stage time elapsed problems.

![Mario Bros](https://github.com/jkaewprateep/multi_agent_queue_action_response/blob/main/Marios%20Bros.gif "Mario Bros")
