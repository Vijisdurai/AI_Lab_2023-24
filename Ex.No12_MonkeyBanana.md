# Ex.No: 11  Planning –  Monkey Banana Problem
### DATE: 3/10/2024                                                                           
### REGISTER NUMBER : 212222220057
### AIM: 
To find the sequence of plan for Monkey Banana problem using PDDL Editor.
###  Algorithm:
Step 1:  Start the program <br> 
Step 2 : Create a domain for Monkey Banana Problem. <br> 
Step 3:  Create a domain by specifying predicates. <br> 
Step 4: Specify the actions GOTO, CLIMB, PUSH-BOX, GET-KNIFE, GRAB-BANANAS in Monkey Banana problem.<br>  
Step 5:   Define a problem for Monkey Banana problem.<br> 
Step 6:  Obtain the plan for given problem.<br> 
Step 7: Stop the program.<br> 
### Program:
```
(define (domain monkey)	       
  (:requirements :strips)
   (:constants monkey box knife bananas glass waterfountain)
   (:predicates (location ?x)
	       (on-floor)
	       (at ?m ?x)
	       (hasknife)
	       (onbox ?x)
	       (hasbananas)
	       (hasglass)
	       (haswater))
   ;; movement and climbing
  (:action GO-TO
	     :parameters (?x ?y)
	     :precondition (and (location ?x) (location ?y) (on-floor) (at monkey ?y))
	     :effect (and (at monkey ?x) (not (at monkey ?y))))
   (:action CLIMB
	     :parameters (?x)
	     :precondition (and (location ?x) (at box ?x) (at monkey ?x))
	     :effect (and (onbox ?x) (not (on-floor))))
   (:action PUSH-BOX
	     :parameters (?x ?y)
	     :precondition (and (location ?x) (location ?y) (at box ?y) (at monkey ?y) 
				 (on-floor))
	     :effect (and (at monkey ?x) (not (at monkey ?y))
			   (at box ?x)    (not (at box ?y))))
  ;; getting bananas
  (:action GET-KNIFE
	     :parameters (?y)
	     :precondition (and (location ?y) (at knife ?y) (at monkey ?y))
	     :effect (and (hasknife) (not (at knife ?y))))
  (:action GRAB-BANANAS
	     :parameters (?y)
	     :precondition (and (location ?y) (hasknife) 
                                 (at bananas ?y) (onbox ?y))
	     :effect (hasbananas))
  ;; getting water
  (:action PICKGLASS
	     :parameters (?y)
	     :precondition (and (location ?y) (at glass ?y) (at monkey ?y))
	     :effect (and (hasglass) (not (at glass ?y))))
  (:action GETWATER
	     :parameters (?y)
	     :precondition (and (location ?y) (hasglass)
				 (at waterfountain ?y)
				 (at monkey ?y)
				 (onbox ?y))
	     :effect (haswater)))
```

### Input 
```

(define (problem pb1)
    	(:domain monkey)
  	(:objects p1 p2 p3 p4 bananas monkey box knife)
  	(:init (location p1)
		(location p2)
		(location p3)
		(location p4)
	 	(at monkey p1)
		(on-floor)
		(at box p2)
		(at bananas p3)
	 	(at knife p4)
	)
  	(:goal (and (hasbananas)))
)
```
### Output/Plan:
Image 1<br>
![image](https://github.com/Ziyavudeen/AI_Lab_2023-24/assets/120120067/22c68f66-aeea-4276-b234-8c9a384c9b11)<br>
Image 2<br>
![image](https://github.com/Ziyavudeen/AI_Lab_2023-24/assets/120120067/9e6547e5-0d20-46e9-98ba-2f81731905ac)<br>
Image 3<br>
![image](https://github.com/Ziyavudeen/AI_Lab_2023-24/assets/120120067/226831c9-3585-4adb-9789-90515feb0a08)<br>
Image 4<br>
![image](https://github.com/Ziyavudeen/AI_Lab_2023-24/assets/120120067/2dc5e07a-8393-46fe-8981-a2e10015a020)<br>
Image 5<br>
![image](https://github.com/Ziyavudeen/AI_Lab_2023-24/assets/120120067/626f454a-c3b9-4099-81f6-1e552cc3aca8)<br>
Image 6<br>
![image](https://github.com/Ziyavudeen/AI_Lab_2023-24/assets/120120067/31537c7c-b273-4360-8f4e-47e8708bc08c)


### Result:
Thus the plan was found for the initial and goal state of given problem.
