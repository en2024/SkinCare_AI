/* toggle menu */

    function toggleFloatingMenu() {
        document.getElementById('fm-btn').classList.toggle('active');
        document.getElementById('fm-overlay').classList.toggle('active');
        document.getElementById('fm-dropdown').classList.toggle('active');
    }

    /* translation labels */

    function applyFloatingMenuLanguage(lang) {
        const isRtl = (lang === 'ar');
        const setHtml = (id, text) => { const el = document.getElementById(id); if (el) el.innerText = text; };
        
        setHtml('fm-title-nav', isRtl ? 'التنقل' : 'Navigation');
        setHtml('fm-nav-home', isRtl ? 'الرئيسية' : 'Home');
        setHtml('fm-nav-catalog', isRtl ? 'الكتالوج' : 'Catalog');
        setHtml('fm-nav-analyzer', isRtl ? 'المحلل' : 'Analyzer');
        setHtml('fm-nav-admin', isRtl ? 'لوحة الإدارة' : 'Admin Dashboard');
        
        setHtml('fm-title-settings', isRtl ? 'الإعدادات' : 'Settings');
        setHtml('fm-set-dark', isRtl ? 'الوضع الليلي' : 'Dark Mode');
        setHtml('fm-set-text', isRtl ? 'حجم الخط' : 'Text Size');
        setHtml('fm-set-lang', isRtl ? 'English / العربية' : 'English / العربية');
        
        setHtml('fm-title-account', isRtl ? 'الحساب' : 'Account');
        setHtml('fm-acc-profile', isRtl ? 'ملفي الشخصي' : 'My Profile');
        setHtml('fm-acc-signout', isRtl ? 'تسجيل خروج' : 'Sign Out');
        setHtml('fm-acc-signin', isRtl ? 'تسجيل دخول' : 'Sign In');
    }

    /* sign out confirm */

    function fmSignOut() {
        const isRtl = document.body.classList.contains('rtl');
        const msg = isRtl ? 'هل أنتِ متأكدة من تسجيل الخروج؟' : 'Are you sure you want to sign out?';
        if (confirm(msg)) {
            window.location.href = "{% url 'logout' %}";
        }
    }

    /* switch language */

    function toggleLanguage() {
        const isCurrentlyRtl = document.body.classList.contains('rtl');
        const newLang = isCurrentlyRtl ? 'en' : 'ar';
        localStorage.setItem('siteLanguage', newLang);
        applyFloatingMenuLanguage(newLang);
        if (typeof applyLanguage === 'function') applyLanguage(newLang);
    }

    /* load saved language */

    (function() {
        const lang = localStorage.getItem('siteLanguage') || 'en';
        if (lang === 'ar') {
            document.body.classList.add('rtl');
            document.documentElement.setAttribute('dir', 'rtl');
            document.documentElement.setAttribute('lang', 'ar');
        }
        
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', function() {
                applyFloatingMenuLanguage(lang);
                if (typeof applyLanguage === 'function') applyLanguage(lang);
            });
        } else {
            applyFloatingMenuLanguage(lang);
            if (typeof applyLanguage === 'function') applyLanguage(lang);
        }
    })();