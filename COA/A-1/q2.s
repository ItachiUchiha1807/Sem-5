.globl main
.data
prompt1 : .asciiz "Enter the first positive integer : "
prompt2 : .asciiz "Enter the second positive integer : "
result : .asciiz"GCD of the two  numbers is : "
newline : .asciiz"\n"
error : .asciiz"Enter positive numbers only!!!"
.text
#$s0=a
#$s1=b

main : 
li $v0,4              #print the the first prompt
la $a0,prompt1
syscall
li $v0,5              #reading a 
syscall
move $s0,$v0

blt $s0,0,negative   #if negative print error message

li $v0,4             #newline 
la $a0,newline
syscall

li $v0,4             # print the second prompt
la $a0,prompt2
syscall
li $v0,5             #reading b
syscall
move $s1,$v0

blt $s1,0,negative   #if negative print error message

beq $s0,0,printb     #if(a==0) print b
j procedure

condition :          #loop condition(b>0)
bgt $s1,0,procedure
beq $s1,0,printa
procedure : 
beq $s1,$s0,printa
blt $s1,$s0,subA     #if(b>a)
blt $s0,$s1,subB     #if(a>b)

subA :               #a=a-b
sub $s0,$s0,$s1      
j condition         #jump to condition

subB :
sub $s1,$s1,$s0      #b=b-a
j condition         #jump to condition 

print :
li $v0,4
la $a0,result       #print the result message
syscall
li $v0,1            #print the result value(GCD)
move $a0,$s0
syscall
li $v0,10           #exit
syscall

negative :          # if  not positive
li $v0,4            #newline
la $a0,newline
syscall 

li $v0,4            # print error message
la $a0,error
syscall
   
j main              #take inputs again

printa :
li $v0,4             #newline 
la $a0,newline
syscall 
li $v0,4
la $a0,result      
syscall             #print out a and exit
li $v0,1
move $a0,$s0
syscall
li $v0,10
syscall

printb :
li $v0,4             #newline 
la $a0,newline
syscall
li $v0,4
la $a0,result       
syscall             #print out b and exit
li $v0,1
move $a0,$s1
syscall
li $v0,10
syscall
exit :
li $v0,10
syscall


 
       
