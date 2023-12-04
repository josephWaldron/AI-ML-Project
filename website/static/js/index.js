function loadFile(event) {
	const image = document.getElementById('inputImage');
	image.src = URL.createObjectURL(event.target.files[0]);
    console.log(image.src)
};
