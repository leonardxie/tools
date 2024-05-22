lsof -i :11001 | awk '{print $2}' | xargs kill -9

lsof -i :11002 | awk '{print $2}' | xargs kill -9

lsof -i :11003 | awk '{print $2}' | xargs kill -9

lsof -i :11004 | awk '{print $2}' | xargs kill -9
