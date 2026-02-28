@echo off
cd C:\Users\TOSHIBA\Desktop\AINC\smartpark
mkdir core
mkdir database
mkdir api
mkdir ui
mkdir demo
mkdir scripts
mkdir data\rules
mkdir data\vehicles
mkdir data\chroma_db
copy /y nul core\__init__.py
copy /y nul database\__init__.py
copy /y nul api\__init__.py
copy /y nul ui\__init__.py
copy /y nul demo\__init__.py
copy /y nul scripts\__init__.py
copy /y nul data\rules\__init__.py
copy /y nul data\vehicles\__init__.py
copy /y nul data\chroma_db\__init__.py
echo done
