alias_to_add="alias reinstall='pip cache purge; pip uninstall -y cudagrad; cd ~/cudagrad; pip install -e .;'"

if [[ -f ~/.bash_profile ]]; then
  echo "$alias_to_add" >> ~/.bash_profile
  echo "Alias added to ~/.bash_profile"

elif [[ -f ~/.bashrc ]]; then
  echo "$alias_to_add" >> ~/.bashrc
  echo "Alias added to ~/.bashrc"

else
  touch ~/.bashrc
  echo "$alias_to_add" >> ~/.bashrc
  echo "Alias added to (a newly created) ~/.bashrc"
fi
