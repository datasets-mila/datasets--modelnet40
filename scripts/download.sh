#!/bin/bash

# this script is meant to be used with 'datalad run'

for file_url in "http://modelnet.cs.princeton.edu/ModelNet40.zip ModelNet40.zip"
do
        echo ${file_url} | git-annex addurl -c annex.largefiles=anything --raw --batch --with-files
done
