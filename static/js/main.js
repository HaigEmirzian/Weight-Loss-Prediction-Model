document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const resultDiv = document.getElementById('result');

    function showResult(message, isError = false) {
        resultDiv.style.display = 'block';
        resultDiv.className = isError ? 'error' : 'success';
        resultDiv.innerHTML = message;
    }

    function showLoading() {
        resultDiv.style.display = 'block';
        resultDiv.className = 'loading';
        resultDiv.innerHTML = 'Processing your data...';
    }

    async function handleSubmit(e) {
        e.preventDefault();
        
        const formData = new FormData();
        const fileInput = document.getElementById('file');

        // Validate file
        if (!fileInput.files || fileInput.files.length === 0) {
            showResult('Please select a file', true);
            return;
        }

        formData.append('file', fileInput.files[0]);
        
        showLoading();
        
        try {
            const response = await fetch('/prediction', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.error) {
                showResult(`Error: ${data.error}`, true);
            } else {
                showResult(`Predicted weight: ${Math.round(data.prediction)} lbs`);
            }
        } catch (error) {
            showResult(`Error: ${error.message}`, true);
        }
    }

    form.addEventListener('submit', handleSubmit);

    // Add file input change handler to show selected filename
    document.getElementById('file').addEventListener('change', function(e) {
        const fileName = e.target.files[0]?.name;
        if (fileName) {
            const label = this.previousElementSibling;
            label.textContent = `Selected file: ${fileName}`;
        }
    });
});
