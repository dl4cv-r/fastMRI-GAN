import argparse


def create_arg_parser(defaults):
    parser = argparse.ArgumentParser()
    parser.set_defaults(**defaults)
    return parser
