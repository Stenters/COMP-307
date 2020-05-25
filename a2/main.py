'''
TODO
    Part2
    Part3
    Part4
    readme
    Report
'''
import pt1, pt2, pt3, pt4

if __name__ == '__main__':
    n = int(input("which part do you want to run? "))
    if (n == 1):
        pt1.run()
    elif (n == 2):
        pt2.run()
    elif (n == 3):
        pt3.run()        
    elif (n == 4):
        pt4.run()
    else:
        pt1.run()        
        pt2.run()        
        pt3.run()
        pt4.run()    
        print("All functions run!")