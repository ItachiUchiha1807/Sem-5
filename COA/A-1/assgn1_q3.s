.data

prompt: .asciiz "Enter a positive integer greater than equals to 10:"
res1: .asciiz "Entered number is a PRIME number"
res2: .asciiz  "entered number is a COMPOSITE number"

.text

main:

li $v0,4		# printing enter postive number message
la $a0,prompt
syscall

li $v0,5
syscall		# $s0 = n
move $s0,$v0

li $s1,2          # i = 2, i = $s1

blt $s0,10,check	#to check the number >= 10

j for 

check:		#to check if n >= 10
li $v0,4		# printing enter postive number message
la $a0,prompt
syscall

li $v0,5
syscall		# $s0 = n
move $s0,$v0

blt $s0,10,check 	#recheck 
j for 		#proceed to for loop


for :
beq $s1,$s0,end1	#exit condition for loop if i == n end

div $s0,$s1		#integer division of n , i
mfhi $s4		#storing remainder of divison in $s4

beq $s4,0,end2	#if remainder = 0 break the loop
add $s1,$s1,1 	#if remainder !=0 continue ; i++;
b for			#loop continues

end1:

li $v0,4		#print the result message
la $a0,res1		#print PRIME
syscall

li $v0,10		#exit main funtion
syscall

end2:			#print the result message
li $v0,4
la $a0,res2		#print COMPOSITE
syscall

li $v0,10		#exit main function	
syscall
