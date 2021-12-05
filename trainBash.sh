while getopts u:a:f: flag
do
    case "${flag}" in
        x) data=${OPTARG};;
        y) labels=${OPTARG};;
    esac
done

#! /usr/bin/python
chmod +x trainBash.py
./trainBash.py data labels