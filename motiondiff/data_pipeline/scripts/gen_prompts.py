
A = ["a person with its left hand carrying a can", "a person with two hands carrying a box", "a person with its right hand carrying a can", "a person who's empty-handed"]
B = ["walks backwards", "stands idle", "repositions to the left", "repositions to the right", "walks in a spiral", "walks in an arc", "turns left", "turns right"]

for a in A:
    for b in B:
        print(a + " " + b)

A = ["a person uses its left hand to pick up a can", "a person uses its right hand to pick up a can", "a person uses its two hands to pick up a box"]
B = ["from the ground", "from a shelf at chest height", "from a shelf at head height", "from a shelf at waist height", "from a shelf at knee height",]

for a in A:
    for b in B:
        print(a + " " + b)

A = ["a person puts down a can with its left hand", "a person puts down a can with its right hand", "a person puts down a box with its two hands"]
B = ["onto the ground", "onto a shelf at chest height", "onto a shelf at head height", "onto a shelf at waist height", "onto a shelf at knee height",]

for a in A:
    for b in B:
        print(a + " " + b)