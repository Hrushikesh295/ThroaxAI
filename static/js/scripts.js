// Slideshow functionality
let currentSlide = 0;
const slides = document.querySelectorAll('.slide');

function showSlide(n) {
    slides.forEach(slide => slide.classList.remove('active'));
    currentSlide = (n + slides.length) % slides.length;
    slides[currentSlide].classList.add('active');
}

function nextSlide() {
    showSlide(currentSlide + 1);
}

// Change slide every 5 seconds
setInterval(nextSlide, 5000);

// Know More button functionality
const knowMoreBtn = document.getElementById('knowMoreBtn');
const hiddenDiseases = document.getElementById('hiddenDiseases');
let showAllDiseases = false;

knowMoreBtn?.addEventListener('click', function() {
    showAllDiseases = !showAllDiseases;
    
    if (showAllDiseases) {
        hiddenDiseases.style.display = 'grid';
        knowMoreBtn.innerHTML = '<span>Show Less</span> <i class="fas fa-chevron-up"></i>';
    } else {
        hiddenDiseases.style.display = 'none';
        knowMoreBtn.innerHTML = '<span>Know More</span> <i class="fas fa-chevron-down"></i>';
    }
    
    // Smooth scroll to expanded section
    if (showAllDiseases) {
        hiddenDiseases.scrollIntoView({ behavior: 'smooth' });
    }
});

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});