python3 main.py --method LR -e 1
python3 main.py --method SVC -e 1
python3 main.py --method XGB -e 1
python3 main.py --method DT -e 1
python3 main.py --method RF -e 1
python3 main.py --method GB -e 1
python3 main.py --method MLP -e 1

python3 main.py --method LR -e 2
python3 main.py --method SVC -e 2
python3 main.py --method XGB -e 2
python3 main.py --method DT -e 2
python3 main.py --method RF -e 2
python3 main.py --method GB -e 2
python3 main_ens.py --method DT GB -e 2