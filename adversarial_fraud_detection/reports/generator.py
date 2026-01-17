"""SR 11-7 compliant PDF report generator."""
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from jinja2 import Environment, FileSystemLoader

# Try to import pdfkit, but make it optional
try:
    import pdfkit
    PDFKIT_AVAILABLE = True
except ImportError:
    PDFKIT_AVAILABLE = False

from src.red_team import RedTeamReport


# Paths
REPORTS_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(REPORTS_DIR, 'template.html')
OUTPUT_DIR = os.path.join(os.path.dirname(REPORTS_DIR), 'outputs', 'reports')

# Feature descriptions for the report
FEATURE_DESCRIPTIONS = {
    'velocity_error': 'Balance equation discrepancy (fraud indicator)',
    'balance_diff': 'Difference between old and new balance',
    'amount': 'Transaction amount in dollars',
    'amount_log': 'Log-transformed transaction amount',
    'hour': 'Hour of day when transaction occurred',
    'oldbalanceOrg': 'Account balance before transaction'
}


class ReportGenerator:
    """
    Generates SR 11-7 compliant model risk assessment reports.

    Supports both HTML and PDF output formats.
    """

    def __init__(self):
        """Initialize the report generator."""
        self.env = Environment(
            loader=FileSystemLoader(REPORTS_DIR),
            autoescape=True
        )
        self.template = self.env.get_template('template.html')
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def _determine_risk_level(self, evasion_rate: float) -> str:
        """Determine overall risk level based on evasion rate."""
        if evasion_rate < 0.05:
            return 'low'
        elif evasion_rate < 0.15:
            return 'medium'
        else:
            return 'high'

    def _generate_executive_summary(
        self,
        attack_results: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> str:
        """Generate executive summary text."""
        evasion_rate = attack_results.get('evasion_rate', 0)
        auc = metrics.get('auc', 0)

        if evasion_rate < 0.05:
            robustness = "demonstrates strong robustness"
            recommendation = "Continue monitoring with quarterly assessments."
        elif evasion_rate < 0.15:
            robustness = "shows moderate vulnerability"
            recommendation = "Consider implementing additional controls and retraining."
        else:
            robustness = "exhibits significant vulnerabilities"
            recommendation = "Immediate remediation required before production deployment."

        return f"""
        The fraud detection model achieved a validation AUC of {auc:.4f}, indicating
        {'excellent' if auc > 0.95 else 'good' if auc > 0.90 else 'moderate'}
        discriminative performance. Adversarial testing with {attack_results.get('total_attacks', 0)}
        attack attempts resulted in an evasion rate of {evasion_rate*100:.1f}%, which
        {robustness} against LLM-generated perturbation attacks. {recommendation}
        """

    def _generate_findings(
        self,
        attack_results: Dict[str, Any],
        feature_importance: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate key findings list."""
        findings = []

        # Evasion rate finding
        evasion_rate = attack_results.get('evasion_rate', 0)
        if evasion_rate < 0.05:
            findings.append(f"Model shows strong adversarial robustness with only {evasion_rate*100:.1f}% evasion rate")
        else:
            findings.append(f"Evasion rate of {evasion_rate*100:.1f}% indicates potential vulnerability to adversarial attacks")

        # Top feature finding
        if feature_importance:
            top_feat = feature_importance[0]
            findings.append(f"Primary fraud indicator is '{top_feat['name']}' with {top_feat['importance']*100:.1f}% importance")

        # Score reduction finding
        avg_reduction = attack_results.get('avg_score_reduction', 0)
        if avg_reduction > 0.1:
            findings.append(f"Average score reduction of {avg_reduction:.3f} suggests attackers can meaningfully reduce detection confidence")
        else:
            findings.append(f"Low average score reduction ({avg_reduction:.3f}) indicates model stability under perturbation")

        # Perturbation success finding
        success_rate = attack_results.get('successful_perturbations', 0) / max(attack_results.get('total_attacks', 1), 1)
        findings.append(f"{success_rate*100:.0f}% of attack attempts generated valid perturbations within constraints")

        return findings

    def _generate_recommendations(
        self,
        attack_results: Dict[str, Any],
        feature_importance: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Generate recommendations based on results."""
        recommendations = []
        evasion_rate = attack_results.get('evasion_rate', 0)

        if evasion_rate >= 0.10:
            recommendations.append({
                'title': 'Adversarial Training',
                'description': 'Incorporate adversarial examples into the training process to improve robustness against evasion attacks.'
            })

        if feature_importance and feature_importance[0]['importance'] > 0.5:
            recommendations.append({
                'title': 'Feature Diversification',
                'description': f"High reliance on '{feature_importance[0]['name']}' ({feature_importance[0]['importance']*100:.0f}%) creates concentration risk. Consider engineering additional features."
            })

        recommendations.append({
            'title': 'Continuous Monitoring',
            'description': 'Implement real-time monitoring for distribution drift and anomalous prediction patterns.'
        })

        recommendations.append({
            'title': 'Periodic Red Team Exercises',
            'description': 'Conduct quarterly adversarial assessments to validate ongoing model robustness.'
        })

        return recommendations

    def generate_html(
        self,
        red_team_report: RedTeamReport,
        model_metrics: Dict[str, Any],
        feature_importance: Dict[str, float],
        shap_plot_path: Optional[str] = None,
        model_version: str = "1.0.0"
    ) -> str:
        """
        Generate HTML report.

        Args:
            red_team_report: Results from red team campaign
            model_metrics: Model performance metrics (auc, precision, recall)
            feature_importance: Feature importance scores
            shap_plot_path: Path to SHAP visualization
            model_version: Model version identifier

        Returns:
            Path to generated HTML file
        """
        # Prepare feature importance data
        feat_list = [
            {
                'name': name,
                'importance': imp,
                'description': FEATURE_DESCRIPTIONS.get(name, 'No description')
            }
            for name, imp in sorted(feature_importance.items(), key=lambda x: -x[1])
        ]

        # Prepare attack results
        attack_results = {
            'total_attacks': red_team_report.total_attacks,
            'successful_perturbations': red_team_report.successful_perturbations,
            'successful_evasions': red_team_report.successful_evasions,
            'evasion_rate': red_team_report.evasion_rate,
            'avg_score_reduction': red_team_report.avg_score_reduction
        }

        # Prepare metrics
        metrics = {
            'auc': f"{model_metrics.get('auc', 0):.4f}",
            'precision': f"{model_metrics.get('precision', 0):.2%}",
            'recall': f"{model_metrics.get('recall', 0):.2%}"
        }

        # Generate dynamic content
        risk_level = self._determine_risk_level(red_team_report.evasion_rate)
        executive_summary = self._generate_executive_summary(attack_results, model_metrics)
        findings = self._generate_findings(attack_results, feat_list)
        recommendations = self._generate_recommendations(attack_results, feat_list)

        # Render template
        html_content = self.template.render(
            report_date=datetime.now().strftime('%Y-%m-%d'),
            model_version=model_version,
            risk_level=risk_level,
            executive_summary=executive_summary,
            metrics=metrics,
            attack_results=attack_results,
            feature_importance=feat_list,
            shap_plot_path=shap_plot_path,
            findings=findings,
            recommendations=recommendations
        )

        # Save HTML
        output_path = os.path.join(
            OUTPUT_DIR,
            f"sr11-7_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return output_path

    def generate_pdf(
        self,
        red_team_report: RedTeamReport,
        model_metrics: Dict[str, Any],
        feature_importance: Dict[str, float],
        shap_plot_path: Optional[str] = None,
        model_version: str = "1.0.0"
    ) -> str:
        """
        Generate PDF report.

        Args:
            red_team_report: Results from red team campaign
            model_metrics: Model performance metrics
            feature_importance: Feature importance scores
            shap_plot_path: Path to SHAP visualization
            model_version: Model version identifier

        Returns:
            Path to generated PDF file
        """
        if not PDFKIT_AVAILABLE:
            print("Warning: pdfkit not available. Install wkhtmltopdf for PDF generation.")
            print("Falling back to HTML output.")
            return self.generate_html(
                red_team_report, model_metrics, feature_importance,
                shap_plot_path, model_version
            )

        # First generate HTML
        html_path = self.generate_html(
            red_team_report, model_metrics, feature_importance,
            shap_plot_path, model_version
        )

        # Convert to PDF
        pdf_path = html_path.replace('.html', '.pdf')

        try:
            pdfkit.from_file(html_path, pdf_path, options={
                'page-size': 'Letter',
                'margin-top': '0.75in',
                'margin-right': '0.75in',
                'margin-bottom': '0.75in',
                'margin-left': '0.75in',
                'encoding': 'UTF-8',
                'enable-local-file-access': None
            })
            print(f"PDF report generated: {pdf_path}")
            return pdf_path
        except Exception as e:
            print(f"PDF generation failed: {e}")
            print(f"HTML report available at: {html_path}")
            return html_path


if __name__ == '__main__':
    from src.models.defender import DefenderModel
    from src.red_team import RedTeamReport

    print("Testing ReportGenerator...")

    # Create mock data
    mock_report = RedTeamReport(
        total_attacks=10,
        successful_perturbations=8,
        successful_evasions=2,
        evasion_rate=0.20,
        avg_score_reduction=0.15
    )

    mock_metrics = {
        'auc': 0.9989,
        'precision': 0.85,
        'recall': 0.90
    }

    mock_importance = {
        'velocity_error': 0.754,
        'balance_diff': 0.118,
        'amount': 0.037,
        'hour': 0.035,
        'oldbalanceOrg': 0.029,
        'amount_log': 0.027
    }

    # Generate report
    generator = ReportGenerator()
    html_path = generator.generate_html(
        mock_report,
        mock_metrics,
        mock_importance
    )

    print(f"\nHTML report generated: {html_path}")
