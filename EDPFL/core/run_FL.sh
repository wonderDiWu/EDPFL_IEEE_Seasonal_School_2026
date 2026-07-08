#python serverrun.py --testbed SM --dataset CIFAR10 --model VGG11 --mode FL &
#sleep 10
#python clientrun.py --testbed SM --dataset CIFAR10 --model VGG11 --ip 127.0.0.1 --index 0 --mode FL &
#python clientrun.py --testbed SM --dataset CIFAR10 --model VGG11 --ip 127.0.0.1 --index 1 --mode FL > /dev/null 2>&1 &
#python clientrun.py --testbed SM --dataset CIFAR10 --model VGG11 --ip 127.0.0.1 --index 2 --mode FL > /dev/null 2>&1 &
#python clientrun.py --testbed SM --dataset CIFAR10 --model VGG11 --ip 127.0.0.1 --index 3 --mode FL > /dev/null 2>&1 &
#python clientrun.py --testbed SM --dataset CIFAR10 --model VGG11 --ip 127.0.0.1 --index 4 --mode FL > /dev/null 2>&1 &


uv run python serverrun.py --testbed PI --dataset CIFAR10 --model LeNet --mode FL
sleep 10
python3 clientrun.py --testbed PI --dataset CIFAR10 --model LeNet --ip 127.0.0.1 --index 0 --mode FL
python3 clientrun.py --testbed PI --dataset CIFAR10 --model LeNet --ip 127.0.0.1 --index 1 --mode FL 