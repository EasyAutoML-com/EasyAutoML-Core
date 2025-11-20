# Security Policy

## 🔒 Supported Versions

We actively support the latest version of EasyAutoML Core. Security updates are provided for:
- Current stable release (latest version)
- Previous stable release (for critical vulnerabilities only)

### Python Version Requirements

- **Python 3.9+** is required
- Keep Python updated to the latest patch version
- Older Python versions may have known security vulnerabilities

### Dependency Versions

See `requirements.txt` for pinned versions of:
- Django 3.2.25
- TensorFlow 2.17
- PyTorch 2.8.0

These versions are selected for stability and security. Update with caution.

## 🚨 Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

### How to Report

If you discover a security vulnerability, please report it responsibly:

1. **Email**: security@easyautoml.com
2. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)
   - Your contact information (if you want credit)

### What to Expect

- **Acknowledgment**: You'll receive an acknowledgment within 48 hours
- **Assessment**: We'll assess the vulnerability within 7 days
- **Updates**: We'll provide regular updates on the status
- **Credit**: We'll credit you in the security advisory (if desired)

### Disclosure Policy

- We'll disclose vulnerabilities after fixes are available
- We'll coordinate disclosure with you
- We'll provide a security advisory with details

## 🔐 Security Best Practices

### For Users

1. **Keep dependencies updated**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Use environment variables** for all secrets
   - Never commit `.env` files
   - Use strong, unique `DJANGO_SECRET_KEY`
   - Secure database credentials

3. **Set `DEBUG=False`** in production
   ```bash
   DEBUG=False
   ```

4. **Configure `ALLOWED_HOSTS`** properly
   ```bash
   ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com
   ```

5. **Use HTTPS** in production
   - Enable SSL/TLS
   - Use secure cookies
   - Configure Django security settings

6. **Keep Python and Django updated**
   - Use supported Python versions
   - Stay on supported Django versions

7. **Review access controls**
   - Limit database access
   - Use strong passwords
   - Implement proper authentication

### For Developers

1. **Never commit secrets**
   - Use environment variables
   - Check `.gitignore` includes `.env`
   - Review commits before pushing

2. **Validate input**
   - Sanitize user input
   - Use Django forms validation
   - Check file uploads

3. **Use prepared statements**
   - Django ORM handles this automatically
   - Avoid raw SQL with user input

4. **Keep dependencies updated**
   - Regularly update requirements
   - Check for known vulnerabilities
   - Use `pip-audit` or similar tools

5. **Review code changes**
   - Check for security implications
   - Test authentication/authorization
   - Verify input validation

## 🛡️ Security Features

### Current Security Measures

- Environment variable configuration (via `python-dotenv`)
- Django security middleware
- CSRF protection
- SQL injection protection (via Django ORM)
- XSS protection (via template escaping)
- Secure password hashing (via Django's password system)
- Input validation on all API endpoints

### Recommended Additions for Production

- Rate limiting for API endpoints (consider `django-ratelimit`)
- Additional input validation and sanitization
- Regular security audits
- Dependency vulnerability scanning (use `pip-audit` or `safety`)
- Web Application Firewall (WAF)
- Intrusion Detection System (IDS)
- Regular penetration testing

## 📋 Security Checklist

Before deploying:

- [ ] `DEBUG=False` in production
- [ ] `DJANGO_SECRET_KEY` is set and secure
- [ ] `ALLOWED_HOSTS` is configured
- [ ] Database credentials are secure
- [ ] HTTPS is enabled
- [ ] Dependencies are updated
- [ ] Security headers are configured
- [ ] `.env` file is not committed

## 🔍 Known Security Considerations

### Environment Variables

- All sensitive data uses environment variables
- Default values are for development only
- Warnings are shown when using defaults

### Database

- Never expose database credentials in code or version control
- Use connection encryption (SSL/TLS) in production
- Limit database user permissions (principle of least privilege)
- Regular backups with encryption
- Monitor for unusual database activity
- Use separate databases for development, testing, and production

### API

- Authentication required for API endpoints
- Rate limiting recommended
- Input validation on all endpoints
- Sanitize file uploads (CSV/Excel files)
- Limit file upload sizes
- Validate data formats before processing

### Machine Learning Security

As an AutoML platform, consider these ML-specific security concerns:

1. **Data Privacy**
   - Sanitize training data before storage
   - Implement data access controls
   - Consider data anonymization for sensitive information
   - Comply with GDPR, CCPA, and other data privacy regulations

2. **Model Security**
   - Protect trained models from unauthorized access
   - Validate model inputs to prevent adversarial attacks
   - Monitor model predictions for anomalies
   - Version control for models

3. **Resource Management**
   - Limit training time and computational resources
   - Implement quotas for users
   - Monitor GPU/CPU usage
   - Prevent resource exhaustion attacks

4. **Input Validation**
   - Validate uploaded datasets (CSV/Excel files)
   - Check for malicious code in data files
   - Limit dataset sizes
   - Sanitize column names and data values

5. **Code Injection Prevention**
   - Never use `eval()` or `exec()` on user input
   - Validate formula strings before execution
   - Sanitize SQL queries (use ORM)
   - Validate JSON data structures

## 📞 Contact

For security issues, please contact:
- **Security Vulnerabilities**: security@easyautoml.com (PRIVATE - do not use GitHub Issues)
- **General Security Questions**: [GitHub Issues](https://github.com/EasyAutoML-com/EasyAutoML-Core/issues) (non-sensitive only)
- **Commercial Support**: legal@easyautoml.com

---

**Last Updated**: 2025

