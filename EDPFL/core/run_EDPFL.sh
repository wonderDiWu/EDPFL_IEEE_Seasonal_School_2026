#python serverrun.py --testbed SM --dataset CIFAR10 --model VGG11 --mode EDPFL &
#sleep 10
#python clientrun.py --testbed SM --dataset CIFAR10 --model VGG11 --ip 127.0.0.1 --index 0 --mode EDPFL &
#python clientrun.py --testbed SM --dataset CIFAR10 --model VGG11 --ip 127.0.0.1 --index 1 --mode EDPFL > /dev/null 2>&1 &
#python clientrun.py --testbed SM --dataset CIFAR10 --model VGG11 --ip 127.0.0.1 --index 2 --mode EDPFL > /dev/null 2>&1 &
#python clientrun.py --testbed SM --dataset CIFAR10 --model VGG11 --ip 127.0.0.1 --index 3 --mode EDPFL > /dev/null 2>&1 &
#python clientrun.py --testbed SM --dataset CIFAR10 --model VGG11 --ip 127.0.0.1 --index 4 --mode EDPFL > /dev/null 2>&1 &


#python serverrun.py --testbed SM --dataset FMNIST --model LeNet --mode EDPFL &
#sleep 10
#python clientrun.py --testbed SM --dataset FMNIST --model LeNet --ip 127.0.0.1 --index 0 --mode EDPFL &
#python clientrun.py --testbed SM --dataset FMNIST --model LeNet --ip 127.0.0.1 --index 1 --mode EDPFL > /dev/null 2>&1 &
#python clientrun.py --testbed SM --dataset FMNIST --model LeNet --ip 127.0.0.1 --index 2 --mode EDPFL > /dev/null 2>&1 &
#python clientrun.py --testbed SM --dataset FMNIST --model LeNet --ip 127.0.0.1 --index 3 --mode EDPFL > /dev/null 2>&1 &
#python clientrun.py --testbed SM --dataset FMNIST --model LeNet --ip 127.0.0.1 --index 4 --mode EDPFL > /dev/null 2>&1 &

python serverrun.py --testbed PI --dataset CIFAR10 --model LeNet --mode EDPFL
sleep 10
python clientrun.py --testbed PI --dataset CIFAR10 --model LeNet --ip 127.0.0.1 --index 0 --mode EDPFL
python clientrun.py --testbed PI --dataset CIFAR10 --model LeNet --ip 127.0.0.1 --index 1 --mode EDPFL 