＃Usage

## Can't find map

If you are doing supervised learning, when opening replay and encounter the following errors:

"Failed to open the map archive: /home/StarCraftII/Battle.net\Cache/fd/88/..."

This is caused by the SC2 cannot find the map, you can execute the following two commands

!wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.1.2.60604_2018_05_16.zip
!unzip -P iagreetotheeula SC2.4.1.2.60604_2018_05_16.zip -d ~ && rm -rf SC2.4.1.2.60604_2018_05_16.zip

This operation will download a lot of maps to the cache at the same time, and then you only need to copy it to the place where the map is not found.