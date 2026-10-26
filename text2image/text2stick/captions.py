"""OFFLINE captions from known dataset annotations; never imported by generation."""
import random

# Phrase families: 0-2 train, 3 wording validation, 4 wording test.
TORSO = [
    [("standing upright","standing straight","keeping the body upright"),
     ("bending forward slightly","leaning forward a little","with a slightly bent body"),
     ("bending forward deeply","leaning far forward","with the body deeply bent forward","bending over","bending at the waist")],
    [("their body is upright","their body is straight","they stand upright"),
     ("their body leans forward a little","they bend forward slightly","their torso is slightly bent"),
     ("their body leans far forward","they bend forward deeply","their torso is deeply bent")],
    [("an upright body","a straight torso","standing with a straight body"),
     ("a slight forward bend","a body leaning slightly forward","a slightly bent torso"),
     ("a deep forward bend","a body leaning deeply forward","a deeply bent torso")],
    [("standing with their torso upright",),("leaning forward with their torso slightly bent",),("leaning forward with their torso deeply bent",)],
    [("keeping their torso straight while standing",),("bending their torso a little forward",),("bending their torso deeply forward",)],
]


def arm(side,state,family,rng):
    choices = [
      [(f"keeping the {side} arm down",f"with the {side} arm hanging down",f"lowering the {side} hand"),
       (f"extending the {side} arm out to the side",f"holding the {side} arm out",f"with the {side} hand stretched sideways",f"stretching the {side} arm sideways"),
       (f"raising the {side} arm",f"holding the {side} hand up",f"with the {side} arm lifted overhead",f"lifting the {side} hand overhead")],
      [(f"their {side} arm is down",f"their {side} hand hangs down"),
       (f"their {side} arm is out to the side",f"their {side} hand is stretched sideways"),
       (f"their {side} arm is raised",f"their {side} hand is up")],
      [(f"the {side} arm lowered",f"the {side} hand hanging down"),
       (f"the {side} arm held sideways",f"the {side} hand extended out to the side"),
       (f"the {side} arm lifted up",f"the {side} hand raised overhead")],
      [(f"their {side} arm hanging down",),(f"their {side} arm extended to the side",),(f"their {side} hand raised up",)],
      [(f"keeping their {side} hand down",),(f"stretching their {side} arm sideways",),(f"lifting their {side} hand overhead",)],
    ]
    return rng.choice(choices[family][state])


def knee(side,state,family,rng):
    choices=[
      [(f"keeping the {side} knee straight",f"with the {side} leg straight"),
       (f"bending the {side} knee",f"with a bent {side} knee")],
      [(f"their {side} knee is straight",f"their {side} leg is straight"),
       (f"their {side} knee is bent",f"their {side} leg is bent at the knee")],
      [(f"a straight {side} leg",f"the {side} knee kept straight"),
       (f"a bent {side} leg",f"the {side} knee bent")],
      [(f"their {side} knee kept straight",),(f"their {side} knee bending",)],
      [(f"keeping their {side} leg straight",),(f"bending their {side} leg at the knee",)],
    ]
    return rng.choice(choices[family][state])


def caption(combo,rng,family=None,partial=False):
    family=rng.randrange(3) if family is None else family
    clauses=[rng.choice(TORSO[family][combo[0]]),arm("left",combo[1],family,rng),
             arm("right",combo[2],family,rng),knee("left",combo[3],family,rng),knee("right",combo[4],family,rng)]
    if partial:
        pick=sorted(rng.sample(range(5),rng.choice([1,2,3])))
        clauses=[clauses[i] for i in pick]
    else:
        # Both-arm expressions are part of training, not a special inference parser.
        if combo[1]==combo[2] and rng.random()<0.3:
            phrase=["with both arms down","with both arms stretched out to the sides","with both hands raised above the head"][combo[1]]
            clauses=[clauses[0],phrase,clauses[3],clauses[4]]
    rng.shuffle(clauses)
    subject=rng.choice(["a person","a stick figure","someone","a man","a woman"])
    if family==0:
        opening=rng.choice(["Draw ","Show ","Please draw ","Make a drawing of ","Could you show "])+subject+" "
        sentence=opening+", ".join(clauses[:-1])+rng.choice([" and "," while "])+clauses[-1] if len(clauses)>1 else opening+clauses[0]
    elif family==1:
        sentence="Draw "+subject+". "+". ".join(c.capitalize() for c in clauses)
    elif family==2:
        sentence=rng.choice(["I want ","A drawing of ","Please show "])+subject+" with "+", ".join(clauses)
    elif family==3:
        sentence="Please make a drawing of "+subject+", "+", ".join(clauses)
    else:
        sentence="Could you draw "+subject+"? "+" and ".join(clauses)
    return sentence.rstrip(".")+"."
