source .env
search_dir=/mnt/data/urls
for entry in $search_dir/*
do
  echo $entry
  python3 downloader/text_extractor.py --file_path $entry
done
