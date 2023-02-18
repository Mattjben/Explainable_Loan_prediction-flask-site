import predictfunc
import re
from predictfunc import ExplainableDecisonTree
print("\n")
print("Welcome to the data janitors risk performance prediction services")
print("\n")
print("Would you like to enter the details manually or use a csv file?")
way=input("Please enter manual/csv: ")

#csv file reading
if way=="csv":
    user_data={}

    print("\n")
    fname=input("Please enter the name of the data file that you want to read the data from: ")
    with open(fname) as fhand:
        for line in fhand:
            lst=line.strip().split(",")
            name=lst[0]
            lst=[int(i) for i in lst[1:]]
            user_data[name]=lst

    print("Thank you, the file is now read.")
    print("\n")

    #looping for more users
    username=input("Please enter the name of the user you want the prediction for: ")
    more=True
    while more:
        features=user_data[username]

        #to be modified
        label="Good"
        edt_array=ExplainableDecisonTree().main(features)
        print(edt_array)


















        print("\n")
        more=input("Would you like to get prediction for any other user?(yes/no): ")
        username=input("Please enter the name of the user you want the prediction for: ")
        while more not in ["yes", "no"]:
            more=input("Would you like to get prediction for any other user?(yes/no): ")
            if more=="yes":
                username=input("Please enter the name of the user you want the prediction for: ")
                more=True
            elif more=="no":
                more=False
            else:
                print("Please enter a valid input from (yes/no)")





##
