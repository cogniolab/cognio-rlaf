# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Currently supported versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take the security of RLAF seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Please DO NOT

- Open a public GitHub issue for security vulnerabilities
- Discuss the vulnerability publicly before it has been addressed

### Please DO

1. **Email us directly** at [moses@cogniolab.com](mailto:moses@cogniolab.com) with:
   - Description of the vulnerability
   - Steps to reproduce the issue
   - Potential impact
   - Suggested fix (if any)

2. **Allow time for response**: We will acknowledge receipt within 48 hours and aim to provide a detailed response within 7 days.

3. **Coordinate disclosure**: We'll work with you to understand and address the issue before any public disclosure.

## Security Best Practices for Users

When using RLAF, follow these security best practices:

### API Key Management

- **Never commit API keys** to version control
- Use environment variables for API keys:
  ```bash
  export ANTHROPIC_API_KEY="your-key-here"
  ```
- Use `.env` files (add to `.gitignore`)
- Consider using secret management tools (AWS Secrets Manager, HashiCorp Vault, etc.)

### Model Safety

- **Review critic feedback** before using in production
- **Implement content filters** for user-facing applications
- **Set rate limits** to prevent abuse
- **Monitor model outputs** for unexpected behavior

### Data Privacy

- **Sanitize training data** before sending to external APIs
- **Review data retention policies** of your LLM providers
- **Implement PII filtering** if handling sensitive data
- **Use local models** for highly sensitive use cases

### Dependency Security

- Regularly update dependencies:
  ```bash
  pip install --upgrade rlaf
  ```
- Monitor security advisories for dependencies
- Use `pip-audit` to check for known vulnerabilities:
  ```bash
  pip install pip-audit
  pip-audit
  ```

### Production Deployment

- **Use authentication** for API endpoints
- **Implement logging** and monitoring
- **Set resource limits** to prevent DoS
- **Regular security audits** of deployed systems
- **Principle of least privilege** for service accounts

## Known Security Considerations

### LLM-Specific Risks

1. **Prompt Injection**: Critics and actors may be vulnerable to prompt injection attacks
   - Sanitize user inputs
   - Use structured prompts
   - Implement input validation

2. **Data Leakage**: Training data may inadvertently leak into model outputs
   - Review training datasets
   - Implement output filtering
   - Use differential privacy techniques when appropriate

3. **Model Poisoning**: Malicious training data could compromise model behavior
   - Validate training data sources
   - Implement data quality checks
   - Use trusted datasets

### API Security

- **Rate limiting**: Implement to prevent abuse
- **Authentication**: Secure all API endpoints
- **Input validation**: Sanitize all user inputs
- **Error handling**: Don't expose sensitive information in error messages

## Security Updates

Security updates will be released as patch versions (e.g., 0.1.1, 0.1.2) and announced via:

- GitHub Security Advisories
- Release notes
- Project README

## Acknowledgments

We appreciate the security research community and will acknowledge reporters (with their permission) in:

- Security advisories
- Release notes
- This SECURITY.md file (Hall of Fame section below)

### Hall of Fame

<!-- Security researchers who have responsibly disclosed vulnerabilities will be listed here -->

_No security vulnerabilities have been reported yet._

## Contact

For security concerns: [moses@cogniolab.com](mailto:moses@cogniolab.com)

For general issues: [GitHub Issues](https://github.com/cogniolab/cognio-rlaf/issues)

---

**Thank you for helping keep RLAF and its users safe!**
