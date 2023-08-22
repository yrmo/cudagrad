python makefile.py bump patch
version=$(python -c "import toml; print(toml.load('pyproject.toml')['project']['version'])")
git tag -a $version -m "Release version $version"
git add cudagrad/__init__.py pyproject.toml
git commit -m $version
git push origin $version
echo "Version $version has been tagged and pushed to GitHub."
file_name="release_notes.txt"
code --wait $file_name
release_notes=$(cat $file_name)
echo "Content captured:"
echo "$release_notes"
gh release create $version -t $version -n "$release_notes"
echo "GitHub release for version $version has been created."
python makefile.py publish
rm $file_name
