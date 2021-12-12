while getopts u:a:f: flag
do
    case "${flag}" in
        x) data=${OPTARG};;
        y) labels=${OPTARG};;
    esac
done

#! /usr/bin/python
chmod +x train_bash.py
./train_bash.py data labels