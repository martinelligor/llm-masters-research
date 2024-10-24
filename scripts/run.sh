make deploy-docker

sleep 3

/bin/bash ${PWD}/scripts/insert_data.sh

make run-streamlit