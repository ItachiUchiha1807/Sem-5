.data

prompt: .asciiz "Enter a positive integer :"
res1: .asciiz "Entered number is perfect number."
res2: .asciiz  "Entered number is a not perfect number."

.text

main:

li $v0,4		# printing enter postive number message
la $a0,prompt
syscall

li $v0,5
syscall		# $s0 = n
move $s0,$v0

li $s1,1          # i = 1, i = $s1
li $s5,0          # sum = 0, sum = $s5
blt $s0,1,check	#to check the number >= 1

j for 

check:		#to check if n >= 1
li $v0,4		# printing enter postive number message
la $a0,prompt
syscall


li $v0,5
syscall		# $s0 = n
move $s0,$v0

blt $s0,1,check 	#recheck 
j for 		#proceed to for loop


for :
beq $s1,$s0,end	#exit condition for loop if i == n end

div $s0,$s1		#integer division of n , i
mfhi $s4		#storing remainder of divison in $s4

beq $s4,0,sum   	#if remainder = 0 add to sum
addi $s1,$s1,1 	#if remainder !=0 continue ; i++;

b for			#loop continues

sum:
add $s5,$s5,$s1
addi $s1,$s1,1
j for


end:

beq $s5,$s0,p1
bne $s5,$s0,p2

p1:
li $v0,4		#print the result message
la $a0,res1		#print Perfect
syscall

li $v0,10		#exit main funtion
syscall

p2:			#print the result message
li $v0,4
la $a0,res2		#print Not Perfect
syscall

li $v0,10		#exit main function	
syscall
