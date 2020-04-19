# TD3 Car Environment 

## Tasks Done to understand the environment:

* **Bullet Ant Environment:**

  [**Notebook**](Reinforcement_learning_Ant_Bullet_Environment_.ipynb)
  
  **Result:**
  
  ![ant_bullet](Assets/ant_bullet.gif)

* **Walker 2D bullet Environment:**

  [**Notebook**](Reinforcement_learning_Walker2D_Bullet_Environment_.ipynb)
  
  **Result:**
  
  ![walker2d_result](Assets/walker_bullet.gif)

* **Half Cheetah Environment:**

  [**Notebook**](Reinforcement_learning_Half_Cheetah_Bullet_Environment_.ipynb)
  
  **Result:**
   
  ![half_cheetah_result](Assets/half_cheetah.gif)

## Failed Attempts

Things I tried to create the custom environment

**1. Created PyGame**

[Game Code](Car_Game)

[Result](Car_Game/game.png)

**2. Integrating with gym environment**

I couldn't integrate it with gym as there is very less documentation, need more time for that. 
I tried also to integrate it directly with gym environemnt.. ended up the same..
most of my time went into that

## Running Attempt:

**Steps:**

* Took Kivy Environment.
* Tried over sensor enviroment.
* Removed A3C and Loaded TD3.
  * Removed Array of rotation and made it to a single continous value given by the model.
  * Added episodes ot the environemnt
  * Added done state:
    * If the car reaches destination
    * If car crosses episodes tiem stamps
 * Click on the lmage below to take you to the video.
 
 * Captured Images of current car location
 

## Need time for:

* Complete the kivy environment.
* COmplete the custom GYM environment.
    
  
